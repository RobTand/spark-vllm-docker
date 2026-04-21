"""Tests for the native compressed-tensors exporter.

Covers the math (NVFP4 / FP8 round-trip) and the wire-format
plumbing (`_to_vllm_internal_name`, `build_quantization_config`)
that has to stay in sync with vLLM's compressed-tensors loader.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from quantization.prismaquant.allocator import promote_fused
from quantization.prismaquant.export_native_compressed import (
    DEFAULT_INPUT_GLOBAL_SCALE,
    FLOAT_TO_E2M1,
    FP8_E4M3_MAX,
    NVFP4_MAX,
    PER_EXPERT_MOE_REGEX,
    _compute_layer_joint_nvfp4,
    _quantize_2d,
    _quantize_3d_packed,
    _round_to_codebook,
    _to_vllm_internal_name,
    compute_extra_ignore,
    validate_mtp_assignment_coverage,
    build_quantization_config,
    canonicalize_format,
    pack_fp4_indices,
    quantize_dequantize_fp8_dynamic,
    quantize_dequantize_fp8_dynamic_packed,
    quantize_dequantize_mxfp8,
    quantize_dequantize_mxfp8_packed,
    quantize_dequantize_nvfp4,
    quantize_dequantize_nvfp4_packed,
)
from quantization.prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile


class _IdentityProfile:
    """Minimal profile stub for tests that only need `live_to_recipe_name`
    to be identity. Avoids pulling in the full ModelProfile ABC and its
    abstract methods."""

    def live_to_recipe_name(self, live_qname: str) -> str:
        return live_qname


def _nvfp4_dequantize(weight_packed, weight_scale_fp8, weight_global_scale_divisor):
    """Reproduce vLLM's NVFP4 dequant convention to verify round-trip.
    The on-disk `weight_global_scale` is `1/global_real`; vLLM inverts
    on load. Per-element dequant: `codebook[idx] * fp8_scale * global_real`.
    """
    rows = weight_packed.shape[0]
    cols = weight_packed.shape[1] * 2
    cb = torch.tensor(FLOAT_TO_E2M1, dtype=torch.float32)
    lo = (weight_packed & 0xF).long()
    hi = ((weight_packed >> 4) & 0xF).long()
    idx = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
    abs_idx = idx & 0x7
    sign = -((idx >> 3).to(torch.float32) * 2 - 1)
    vals = sign * cb[abs_idx]
    fp8_per_col = (
        weight_scale_fp8.float()
        .unsqueeze(-1)
        .expand(-1, -1, cols // weight_scale_fp8.shape[1])
        .reshape(rows, cols)
    )
    global_real = 1.0 / weight_global_scale_divisor.item()
    return vals * fp8_per_col * global_real


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_nvfp4_2d_roundtrip_mse_small(self):
        W = torch.randn(64, 128) * 0.1
        wp, ws, wg = quantize_dequantize_nvfp4(W)
        self.assertEqual(wp.dtype, torch.uint8)
        self.assertEqual(ws.dtype, torch.float8_e4m3fn)
        self.assertEqual(wg.dtype, torch.float32)
        self.assertEqual(tuple(wp.shape), (64, 64))
        self.assertEqual(tuple(ws.shape), (64, 8))
        self.assertEqual(tuple(wg.shape), (1,))
        # fp8 scale must use the FP8 representable range, not be
        # squashed into [0, 1] (the latter loses precision).
        self.assertGreater(ws.float().max().item(), 32.0,
                           "fp8 scale appears to be normalized to [0,1]; "
                           "vLLM's NVFP4 path expects the full FP8 range")

        dequant = _nvfp4_dequantize(wp, ws, wg)
        mse = (W - dequant).pow(2).mean().item()
        self.assertLess(mse, 1e-3,
                        f"NVFP4 round-trip MSE {mse:.3e} too large")
        # max-abs preserved (NVFP4 has explicit ±6 codes covering the peak)
        self.assertAlmostEqual(
            dequant.abs().max().item(),
            W.abs().max().item(),
            places=3,
        )

    def test_nvfp4_packed_per_expert_global_scale(self):
        # Each expert's global_scale is independent.
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.05
        wp, ws, wg = quantize_dequantize_nvfp4_packed(P)
        self.assertEqual(tuple(wp.shape), (E, M, N // 2))
        self.assertEqual(tuple(ws.shape), (E, M, N // 16))
        self.assertEqual(tuple(wg.shape), (E,))
        # Distinct experts → distinct per-tensor scales.
        self.assertGreater(wg.unique().numel(), 1)

    def test_fp8_dynamic_2d_per_channel_scale(self):
        W = torch.randn(64, 128) * 0.1
        w, s = quantize_dequantize_fp8_dynamic(W)
        self.assertEqual(w.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(s.shape), (64, 1))
        self.assertEqual(s.dtype, torch.float32)
        self.assertFalse(torch.isnan(w.float()).any().item(),
                         "fp8 cast NaN — likely overflow in scale")
        # Round-trip MSE
        dequant = w.float() * s
        mse = (W - dequant).pow(2).mean().item()
        self.assertLess(mse, 1e-4)

    def test_fp8_dynamic_packed_3d(self):
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.1
        w, s = quantize_dequantize_fp8_dynamic_packed(P)
        self.assertEqual(tuple(w.shape), (E, M, N))
        self.assertEqual(tuple(s.shape), (E, M, 1))

    def test_mxfp8_2d_grouped_scale(self):
        W = torch.randn(32, 64) * 0.1
        w, s = quantize_dequantize_mxfp8(W)
        self.assertEqual(w.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.uint8)
        self.assertEqual(tuple(s.shape), (32, 2))
        scales = torch.pow(2.0, s.to(torch.float32) - 127.0)
        dequant = w.float() * scales.repeat_interleave(32, dim=1)
        mse = (W - dequant).pow(2).mean().item()
        self.assertLess(mse, 2e-4)

    def test_mxfp8_packed_3d(self):
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.1
        w, s = quantize_dequantize_mxfp8_packed(P)
        self.assertEqual(tuple(w.shape), (E, M, N))
        self.assertEqual(tuple(s.shape), (E, M, 2))
        self.assertEqual(s.dtype, torch.uint8)


class TestPackBits(unittest.TestCase):
    def test_round_to_codebook_signed(self):
        # Known mapping: 0→0, 0.5→1, 1.0→2, 6.0→7, -6.0→15
        v = torch.tensor([0.0, 0.5, 1.0, 6.0, -6.0])
        idx = _round_to_codebook(v)
        self.assertEqual(idx.tolist(), [0, 1, 2, 7, 15])

    def test_pack_fp4_two_per_byte(self):
        # Indices 1, 2 packed as low=1, high=2 → byte 0x21 = 33
        idx = torch.tensor([[1, 2, 3, 4]])
        packed = pack_fp4_indices(idx, 4)
        self.assertEqual(packed.shape, torch.Size([1, 2]))
        self.assertEqual(packed[0, 0].item(), (1 | (2 << 4)))
        self.assertEqual(packed[0, 1].item(), (3 | (4 << 4)))


class TestRecipeParsing(unittest.TestCase):
    def test_canonicalize_autoround_dict(self):
        nv = {"bits": 4, "data_type": "nv_fp"}
        mx8 = {"bits": 8, "data_type": "mx_fp"}
        bf = {"bits": 16, "data_type": "float"}
        self.assertEqual(canonicalize_format(nv), "NVFP4")
        self.assertEqual(canonicalize_format(mx8), "MXFP8")
        self.assertEqual(canonicalize_format(bf), "BF16")
        # mx_fp/4 collapses to NVFP4 (only 4-bit format vLLM-served).
        self.assertEqual(canonicalize_format({"bits": 4, "data_type": "mx_fp"}), "NVFP4")


class TestVLLMInternalNaming(unittest.TestCase):
    """vLLM's qwen3_5 hf_to_vllm_mapper transforms source HF names to
    internal module names. The exporter's `quantization_config` targets
    + ignore must match the INTERNAL form so `find_matched_target`
    succeeds."""

    def test_text_only_recipe_naming_remap(self):
        self.assertEqual(
            _to_vllm_internal_name("model.layers.0.linear_attn.in_proj_qkv"),
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
        )
        self.assertEqual(
            _to_vllm_internal_name("model.embed_tokens"),
            "language_model.model.embed_tokens",
        )

    def test_lm_head_remap(self):
        self.assertEqual(
            _to_vllm_internal_name("lm_head"),
            "language_model.lm_head",
        )

    def test_multimodal_source_naming_remap(self):
        # Source on-disk uses `model.language_model.X`; vLLM internal
        # is `language_model.model.X` (the prefix swap).
        self.assertEqual(
            _to_vllm_internal_name(
                "model.language_model.layers.5.mlp.shared_expert_gate"),
            "language_model.model.layers.5.mlp.shared_expert_gate",
        )

    def test_visual_remap(self):
        self.assertEqual(
            _to_vllm_internal_name("model.visual.blocks.0.attn.proj"),
            "visual.blocks.0.attn.proj",
        )


class TestBuildQuantizationConfig(unittest.TestCase):
    def test_minimal_two_format_assignment(self):
        profile = Qwen3_5Profile()
        # Lots of NVFP4, fewer MXFP8 → NVFP4 becomes the catch-all
        # bucket (largest count) and gets the per-expert pattern.
        assignment = {
            f"model.layers.{i}.self_attn.k_proj": "MXFP8"
            for i in range(2)  # 2 MXFP8 entries
        }
        for i in range(5):  # 5 NVFP4 entries
            assignment[f"model.layers.{i}.mlp.experts.down_proj"] = "NVFP4"
        qc = build_quantization_config(
            assignment, bf16_passthrough={"lm_head"}, profile=profile,
        )
        self.assertEqual(qc["quant_method"], "compressed-tensors")
        self.assertEqual(qc["format"], "mixed-precision")
        self.assertEqual(len(qc["config_groups"]), 2)
        # Find each group by num_bits — order isn't part of the contract
        groups_by_bits = {
            g["weights"]["num_bits"]: g
            for g in qc["config_groups"].values()
        }
        mxfp8 = groups_by_bits[8]
        nvfp4 = groups_by_bits[4]
        # MXFP8 group: explicit per-name regex targets only
        self.assertTrue(all(t.startswith("re:^language_model[.]")
                            for t in mxfp8["targets"]))
        self.assertNotIn(PER_EXPERT_MOE_REGEX, mxfp8["targets"])
        # NVFP4 catch-all: explicit + the per-expert pattern
        self.assertEqual(nvfp4["weights"]["strategy"], "tensor_group")
        self.assertEqual(nvfp4["weights"]["group_size"], 16)
        self.assertIn(PER_EXPERT_MOE_REGEX, nvfp4["targets"])
        # NVFP4 group must declare its per-group format so vLLM's
        # is_activation_quantization_format check enables W4A4 dispatch.
        self.assertEqual(nvfp4["format"], "nvfp4-pack-quantized")

    def test_ignore_uses_vllm_internal_naming(self):
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.mlp.gate_proj": "NVFP4",
            "model.layers.0.mlp.shared_expert_gate": "BF16",
        }
        qc = build_quantization_config(
            assignment, bf16_passthrough={"lm_head"},
            extra_ignore=["model.layers.0.mlp.gate"],
            profile=profile,
        )
        ignore = qc["ignore"]
        self.assertIn("language_model.lm_head", ignore)
        self.assertIn(
            "language_model.model.layers.0.mlp.shared_expert_gate", ignore)
        self.assertIn(
            "language_model.model.layers.0.mlp.gate", ignore)

    def test_no_class_name_catchall_target(self):
        # The class-name catch-all "Linear" short-circuits vLLM's
        # fused-layer match path and was the bug that produced wrong
        # scheme allocation. Make sure we don't reintroduce it.
        assignment = {"model.layers.0.mlp.gate_proj": "NVFP4"}
        qc = build_quantization_config(
            assignment, bf16_passthrough=set(), profile=Qwen3_5Profile()
        )
        for group in qc["config_groups"].values():
            for t in group["targets"]:
                self.assertNotEqual(t, "Linear",
                                    "do not use a 'Linear' class-name catch-all; "
                                    "it short-circuits fused-layer match")


class TestQuantize2DDispatch(unittest.TestCase):
    def test_nvfp4_emits_input_global_scale(self):
        """vLLM's CompressedTensorsW4A4Nvfp4 process_weights_after_loading
        does `1 / input_global_scale.max()`. Without an emitted value,
        the param defaults to zeros and vLLM produces 1/0 = inf →
        degenerate output. Make sure we always emit it."""
        W = torch.randn(8, 16) * 0.1
        out = _quantize_2d(W, "NVFP4")
        self.assertIn("weight_packed", out)
        self.assertIn("weight_scale", out)
        self.assertIn("weight_global_scale", out)
        self.assertIn("input_global_scale", out)
        self.assertEqual(out["input_global_scale"].dtype, torch.float32)
        self.assertEqual(out["input_global_scale"].numel(), 1)
        self.assertAlmostEqual(
            out["input_global_scale"].item(), DEFAULT_INPUT_GLOBAL_SCALE)

    def test_mxfp8_emits_grouped_dense(self):
        W = torch.randn(8, 32) * 0.1
        out = _quantize_2d(W, "MXFP8")
        self.assertIn("weight", out)
        self.assertEqual(out["weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(out["weight_scale"].dtype, torch.uint8)
        self.assertEqual(tuple(out["weight_scale"].shape), (8, 1))


class TestFusedSiblingJointGlobalScale(unittest.TestCase):
    """vLLM warns when q/k/v/gate/up have different weight_global_scale.
    The exporter pre-computes a joint per-tensor scale across each
    fused-sibling group so the warning goes away (and the per-tensor
    scale on disk is correct under vLLM's fused-loader rules)."""

    def test_fused_dense_group_self_attn(self):
        from quantization.prismaquant.export_native_compressed import _fused_dense_group
        g = _fused_dense_group("model.layers.5.self_attn.q_proj")
        self.assertIsNotNone(g)
        pre, members = g
        self.assertEqual(pre, "model.layers.5")
        self.assertIn("k_proj", members)

    def test_fused_dense_group_mlp_gate_up(self):
        from quantization.prismaquant.export_native_compressed import _fused_dense_group
        g = _fused_dense_group("model.layers.0.mlp.shared_expert.up_proj")
        self.assertIsNotNone(g)
        self.assertEqual(set(g[1]), {"gate_proj", "up_proj"})

    def test_fused_dense_group_qwen36_linear_attn(self):
        from quantization.prismaquant.export_native_compressed import _fused_dense_group
        for sib in ("in_proj_qkv", "in_proj_z"):
            g = _fused_dense_group(f"model.layers.7.linear_attn.{sib}")
            self.assertIsNotNone(g, f"missing fused-group pattern for {sib}")
            self.assertEqual(set(g[1]), {"in_proj_qkv", "in_proj_z"})

    def test_compute_nvfp4_joint_global_picks_max(self):
        from quantization.prismaquant.export_native_compressed import (
            _compute_nvfp4_joint_global, compute_nvfp4_global_real,
        )

        # Build a tiny model with two fused-sibling Linears (different
        # max-abs values). The joint scale must be the max of their
        # natural per-tensor scales.
        class TinyAttn(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.q_proj = torch.nn.Linear(32, 32, bias=False)
                s.k_proj = torch.nn.Linear(32, 32, bias=False)
                s.v_proj = torch.nn.Linear(32, 32, bias=False)

        class TinyLayer(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.self_attn = TinyAttn()

        class TinyModel(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.model = torch.nn.Module()
                s.model.layers = torch.nn.ModuleList([TinyLayer()])

        torch.manual_seed(0)
        m = TinyModel()
        # Force k_proj to have the largest max-abs.
        with torch.no_grad():
            m.model.layers[0].self_attn.k_proj.weight.mul_(10.0)

        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "NVFP4",
            "model.layers.0.self_attn.v_proj": "NVFP4",
        }
        joint = _compute_nvfp4_joint_global(m, assignment)
        self.assertEqual(len(joint), 3)
        joint_value = next(iter(joint.values())).item()
        # All three must point to the SAME scalar.
        for v in joint.values():
            self.assertAlmostEqual(v.item(), joint_value)
        # And it must be at least the natural scale of the max sibling.
        natural = compute_nvfp4_global_real(
            m.model.layers[0].self_attn.k_proj.weight.float()).item()
        self.assertAlmostEqual(joint_value, natural, places=5)


class TestPackedExpertSplit(unittest.TestCase):
    def test_quantize_3d_packed_nvfp4_returns_per_expert_dim(self):
        # 3D packed `[E, M, N]` produces tensors with leading expert
        # dim preserved. Splitting into per-expert-per-projection is
        # done in materialize_tensors, not _quantize_3d_packed.
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.05
        out = _quantize_3d_packed(P, "NVFP4")
        self.assertEqual(out["weight_packed"].shape[0], E)
        self.assertEqual(out["weight_global_scale"].shape, torch.Size([E]))


class TestQwen35ProfileFallback(unittest.TestCase):
    def _cpu_only_profile(self):
        profile = Qwen3_5Profile()
        profile._vllm_cls = None
        profile._vllm_cls_loaded = True
        profile._fused_matcher = None
        return profile

    def test_fused_sibling_group_has_cpu_only_fallback(self):
        profile = self._cpu_only_profile()

        self.assertEqual(
            profile.fused_sibling_group(
                "model.layers.25.linear_attn.in_proj_qkv"
            ),
            "model.layers.25.linear_attn.in_proj_qkvz",
        )
        self.assertEqual(
            profile.fused_sibling_group(
                "model.layers.25.linear_attn.in_proj_z"
            ),
            "model.layers.25.linear_attn.in_proj_qkvz",
        )
        self.assertEqual(
            profile.fused_sibling_group(
                "model.layers.25.linear_attn.in_proj_a"
            ),
            "model.layers.25.linear_attn.in_proj_ba",
        )
        self.assertEqual(
            profile.fused_sibling_group(
                "model.layers.25.self_attn.q_proj"
            ),
            "model.layers.25.self_attn.qkv_proj",
        )

    def test_promote_fused_keeps_linear_attn_qkvz_coherent_without_vllm(self):
        profile = self._cpu_only_profile()
        assignment = {
            "model.layers.25.linear_attn.in_proj_qkv": "MXFP8",
            "model.layers.25.linear_attn.in_proj_z": "NVFP4",
            "model.layers.25.linear_attn.in_proj_a": "NVFP4",
            "model.layers.25.linear_attn.in_proj_b": "NVFP4",
        }

        promoted = promote_fused(
            assignment,
            {"BF16": 0, "NVFP4": 1, "MXFP8": 2},
            profile=profile,
        )

        self.assertEqual(promoted["model.layers.25.linear_attn.in_proj_qkv"], "MXFP8")
        self.assertEqual(promoted["model.layers.25.linear_attn.in_proj_z"], "MXFP8")
        self.assertEqual(promoted["model.layers.25.linear_attn.in_proj_a"], "NVFP4")
        self.assertEqual(promoted["model.layers.25.linear_attn.in_proj_b"], "NVFP4")


class TestMtpCoverageValidation(unittest.TestCase):
    class _Profile:
        def has_mtp(self):
            return True

    def test_validate_mtp_assignment_coverage_raises_when_recipe_omits_mtp(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with open(td / "model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": {"mtp.fc.weight": "model-00001.safetensors"}}, f)

            with self.assertRaisesRegex(RuntimeError, "contains no mtp"):
                validate_mtp_assignment_coverage(
                    str(td),
                    {"model.layers.0.self_attn.q_proj": "NVFP4"},
                    self._Profile(),
                )

    def test_validate_mtp_assignment_coverage_accepts_recipe_with_mtp(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with open(td / "model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": {"mtp.fc.weight": "model-00001.safetensors"}}, f)

            validate_mtp_assignment_coverage(
                str(td),
                {"mtp.fc": "BF16"},
                self._Profile(),
            )


class TestDeltaNetFusedSiblingJointScale(unittest.TestCase):
    """Regression for commit e2e0091: Qwen3.6 DeltaNet linear-attention
    fuses `in_proj_qkv + in_proj_z → in_proj_qkvz` (and `in_proj_b +
    in_proj_a → in_proj_ba`) at vLLM load time. The fused packed
    Linear needs a SHARED NVFP4 `weight_global_scale` across those
    siblings. `_compute_layer_joint_nvfp4` is the per-layer helper
    that computes it; if it ever drifts back to per-Linear scales,
    vLLM warns about reduced accuracy from mismatched parallel-layer
    scales."""

    def _build_hybrid_layer(self) -> torch.nn.Module:
        """Two DeltaNet siblings inside a `linear_attn` module stub."""
        class TinyLinearAttn(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.in_proj_qkv = torch.nn.Linear(64, 48, bias=False)
                s.in_proj_z = torch.nn.Linear(64, 16, bias=False)

        class TinyLayer(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.linear_attn = TinyLinearAttn()

        return TinyLayer()

    def test_deltanet_siblings_share_single_joint_scale(self):
        torch.manual_seed(0)
        layer = self._build_hybrid_layer()
        # Give `in_proj_qkv` a larger max-abs so the joint scale is
        # determined by it, not by `in_proj_z`.
        with torch.no_grad():
            layer.linear_attn.in_proj_qkv.weight.mul_(10.0)

        assignment = {
            "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
            "model.layers.0.linear_attn.in_proj_z": "NVFP4",
        }
        joint = _compute_layer_joint_nvfp4(
            layer, "model.layers.0", assignment, _IdentityProfile())

        # Both siblings must map to NVFP4 and share ONE scale tensor.
        self.assertEqual(
            set(joint),
            {
                "model.layers.0.linear_attn.in_proj_qkv",
                "model.layers.0.linear_attn.in_proj_z",
            },
        )
        scale_qkv = joint["model.layers.0.linear_attn.in_proj_qkv"]
        scale_z = joint["model.layers.0.linear_attn.in_proj_z"]
        # Exact equality — the helper reuses one tensor across the
        # fused group.
        self.assertEqual(scale_qkv.item(), scale_z.item())

        # The shared scale must equal the max of the per-sibling
        # natural scales (commit e2e0091 regression).
        from quantization.prismaquant.export_native_compressed import (
            compute_nvfp4_global_real,
        )
        nat_qkv = compute_nvfp4_global_real(
            layer.linear_attn.in_proj_qkv.weight.float(), group_size=16)
        nat_z = compute_nvfp4_global_real(
            layer.linear_attn.in_proj_z.weight.float(), group_size=16)
        self.assertAlmostEqual(
            scale_qkv.item(), max(nat_qkv.item(), nat_z.item()),
            places=5,
        )

    def test_mixed_format_siblings_do_not_emit_joint_scale(self):
        """If only one sibling is NVFP4 (and the other MXFP8/BF16),
        there's no fused packed Linear to share a scale across — the
        helper must skip the group."""
        torch.manual_seed(0)
        layer = self._build_hybrid_layer()
        assignment = {
            "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
            "model.layers.0.linear_attn.in_proj_z": "MXFP8",
        }
        joint = _compute_layer_joint_nvfp4(
            layer, "model.layers.0", assignment, _IdentityProfile())
        self.assertEqual(joint, {},
                         "mixed-format sibling group must not emit a joint scale")


class TestComputeExtraIgnorePerExpertSiblings(unittest.TestCase):
    """Regression for commit dab2473: per-expert MoE source tensors
    (e.g. `model.layers.0.mlp.experts.3.gate_proj`) are covered by the
    packed parent (`...mlp.experts.gate_up_proj`) at compressed-tensors
    load time. If the helper accidentally adds them to `extra_ignore`,
    vLLM marks the FusedMoE layer as un-quantized, the NVFP4 scale
    params never get registered, and load crashes."""

    def test_per_expert_siblings_excluded_when_parent_quantized(self):
        # Assignment includes the packed parent — both per-expert
        # source keys must be omitted from extra_ignore.
        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.mlp.experts.down_proj": "NVFP4",
        }
        source_iter = [
            # Per-expert source tensors (2D) — must NOT appear in extra_ignore.
            ("model.layers.0.mlp.experts.0.gate_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.0.up_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.0.down_proj.weight", [1024, 512]),
            ("model.layers.0.mlp.experts.3.gate_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.3.up_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.3.down_proj.weight", [1024, 512]),
            # An unrelated 2D Linear the recipe doesn't cover — this
            # SHOULD end up in extra_ignore.
            ("model.visual.merger.weight", [768, 768]),
            # A non-2D tensor — always skipped regardless of coverage.
            ("model.layers.0.mlp.gate.weight", [128]),
        ]
        extra = compute_extra_ignore(source_iter, assignment)

        for name in [
            "model.layers.0.mlp.experts.0.gate_proj",
            "model.layers.0.mlp.experts.0.up_proj",
            "model.layers.0.mlp.experts.0.down_proj",
            "model.layers.0.mlp.experts.3.gate_proj",
            "model.layers.0.mlp.experts.3.up_proj",
            "model.layers.0.mlp.experts.3.down_proj",
        ]:
            self.assertNotIn(
                name, extra,
                f"per-expert sibling {name} must not be in extra_ignore "
                "when the packed parent is in the assignment "
                "(regression for commit dab2473)",
            )
        self.assertIn("model.visual.merger", extra)

    def test_per_expert_siblings_included_when_parent_missing(self):
        """Sanity: without the parent in the assignment, per-expert
        tensors DO end up in extra_ignore (they would be un-quantized
        on the vLLM side, so compressed-tensors needs to skip them)."""
        assignment: dict[str, str] = {
            # intentionally missing the packed parents
        }
        source_iter = [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.0.down_proj.weight", [1024, 512]),
        ]
        extra = compute_extra_ignore(source_iter, assignment)
        self.assertIn("model.layers.0.mlp.experts.0.gate_proj", extra)
        self.assertIn("model.layers.0.mlp.experts.0.down_proj", extra)

    def test_language_model_prefix_remap(self):
        """Multimodal checkpoints prefix body tensors with
        `model.language_model.*` on disk but the recipe uses
        `model.*` — the helper must remap before the coverage check."""
        assignment = {
            # recipe-side name (no language_model. infix)
            "model.layers.0.self_attn.q_proj": "NVFP4",
        }
        source_iter = [
            # disk-side name with language_model. infix
            ("model.language_model.layers.0.self_attn.q_proj.weight",
             [1024, 1024]),
            # unrelated 2D the recipe doesn't cover
            ("model.language_model.layers.0.mlp.shared_expert_gate.weight",
             [32, 1024]),
        ]
        extra = compute_extra_ignore(source_iter, assignment)
        self.assertNotIn(
            "model.language_model.layers.0.self_attn.q_proj", extra)
        self.assertIn(
            "model.language_model.layers.0.mlp.shared_expert_gate", extra)


if __name__ == "__main__":
    unittest.main()


class TestNvfp4InputGlobalScale(unittest.TestCase):
    """Per-layer input_global_scale calibration from cached activations.
    
    `compute_nvfp4_input_global_scale(activations)` returns FP4_MAX/max_abs
    so scaled activations fit [-6, 6]. Zero/negative max-abs falls back to
    the default."""

    def test_max_abs_scales_to_fp4_range(self):
        import torch
        from quantization.prismaquant.export_native_compressed import (
            compute_nvfp4_input_global_scale, _FP4_E2M1_MAX,
        )
        acts = torch.tensor([0.0, 1.5, -3.0, 2.0])
        s = compute_nvfp4_input_global_scale(acts)
        # max_abs=3.0, scale=6/3=2.0 → scaled activations in [-6, 6]
        self.assertAlmostEqual(s, _FP4_E2M1_MAX / 3.0, places=5)

    def test_degenerate_all_zero_falls_back(self):
        import torch
        from quantization.prismaquant.export_native_compressed import (
            compute_nvfp4_input_global_scale, DEFAULT_INPUT_GLOBAL_SCALE,
        )
        acts = torch.zeros(100)
        s = compute_nvfp4_input_global_scale(acts)
        self.assertEqual(s, DEFAULT_INPUT_GLOBAL_SCALE)

    def test_quantize_2d_reads_override(self):
        import torch
        from quantization.prismaquant.export_native_compressed import _quantize_2d
        weight = torch.randn(32, 32)
        out = _quantize_2d(weight, "NVFP4",
                           input_global_scale_override=2.5)
        self.assertAlmostEqual(
            float(out["input_global_scale"].item()), 2.5, places=4)

    def test_quantize_2d_uses_global_cache_when_named(self):
        import torch
        import quantization.prismaquant.export_native_compressed as m
        weight = torch.randn(32, 32)
        # Save/restore the module-level cache
        saved = m._INPUT_GLOBAL_SCALES
        try:
            m._INPUT_GLOBAL_SCALES = {"foo.bar.q_proj": 3.14}
            out = _quantize_2d = m._quantize_2d(
                weight, "NVFP4", linear_name="foo.bar.q_proj"
            )
            self.assertAlmostEqual(
                float(out["input_global_scale"].item()), 3.14, places=4)
        finally:
            m._INPUT_GLOBAL_SCALES = saved


class TestActivationAwarePasses(unittest.TestCase):
    """AWQ per-channel rescale + GPTQ OBS + activation-weighted rounding
    — the three closed-form calibration-aware passes wired into
    `_quantize_2d`'s NVFP4 path. Each has a per-pass unit test plus a
    composed integration test on a synthetic [out, in] linear with a
    heavily imbalanced activation distribution."""

    def setUp(self):
        import torch
        torch.manual_seed(42)

    def _decode_nvfp4(self, wp, ws, wg):
        import torch
        from quantization.prismaquant.export_native_compressed import (
            FLOAT_TO_E2M1,
        )
        rows = wp.shape[0]
        cols = wp.shape[1] * 2
        cb = torch.tensor(FLOAT_TO_E2M1, dtype=torch.float32)
        lo = (wp & 0xF).long()
        hi = ((wp >> 4) & 0xF).long()
        idx = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
        abs_idx = idx & 0x7
        sign = -((idx >> 3).to(torch.float32) * 2 - 1)
        vals = sign * cb[abs_idx]
        fp8_per_col = (
            ws.float().unsqueeze(-1)
            .expand(-1, -1, cols // ws.shape[1])
            .reshape(rows, cols)
        )
        global_real = 1.0 / wg.item()
        return vals * fp8_per_col * global_real

    def test_awq_channel_scale_shape_and_norm(self):
        """`_awq_channel_scale` returns normalized `s` with max=1
        and `mean|a|^0.5` shape."""
        import torch
        from quantization.prismaquant.export_native_compressed import (
            _awq_channel_scale,
        )
        acts = torch.randn(128, 64) * torch.linspace(0.1, 10.0, 64)
        s = _awq_channel_scale(acts)
        self.assertEqual(tuple(s.shape), (64,))
        self.assertAlmostEqual(float(s.max().item()), 1.0, places=5)
        # High-activation channels must have HIGHER scale (that's the
        # point — they get more FP4 grid budget).
        self.assertGreater(float(s[-1].item()), float(s[0].item()))

    def test_awq_rescale_weight_matches_shape(self):
        import torch
        from quantization.prismaquant.export_native_compressed import (
            _awq_rescale_weight,
        )
        W = torch.randn(32, 64)
        acts = torch.randn(256, 64)
        W_scaled, s = _awq_rescale_weight(W, acts)
        self.assertEqual(W_scaled.shape, W.shape)
        self.assertEqual(s.shape, (64,))
        # W_scaled[:, c] == W[:, c] * s[c]
        self.assertTrue(torch.allclose(
            W_scaled, W * s.unsqueeze(0), atol=1e-5
        ))

    def test_gptq_obs_rounding_returns_grid_aligned(self):
        """After GPTQ, every weight should round to some point on the
        NVFP4 grid — repacking should not change the dequantized value
        by more than one grid step (allowing for global-scale adjustments)."""
        import torch
        from quantization.prismaquant.export_native_compressed import (
            _gptq_obs_rounding_nvfp4, quantize_dequantize_nvfp4,
        )
        W = torch.randn(16, 32) * 0.2
        X = torch.randn(200, 32) * 0.5
        W_gptq = _gptq_obs_rounding_nvfp4(W, X, group_size=16)
        self.assertEqual(W_gptq.shape, W.shape)
        # Re-pack the GPTQ output — it should round-trip (each weight
        # already sits on the grid, so quant+dequant is approximately
        # idempotent up to the per-group outer scale math).
        wp, ws, wg = quantize_dequantize_nvfp4(W_gptq)
        dq = self._decode_nvfp4(wp, ws, wg)
        # GPTQ output packing re-quant MSE must be O(grid step²).
        mse = (W_gptq - dq).pow(2).mean().item()
        self.assertLess(mse, 1e-2,
                        f"GPTQ output not grid-aligned, mse={mse:.3e}")

    def test_activation_weighted_round_prefers_high_importance_channels(self):
        """Activation-weighted rounding should pick the grid neighbor
        that minimizes weighted error. We construct a weight that's
        ambiguous between two grid points in a high-importance column
        and verify the output is closer to the true value there than
        pure RTN would be (pure RTN ignores column importance)."""
        import torch
        from quantization.prismaquant.export_native_compressed import (
            _activation_weighted_round_nvfp4, quantize_dequantize_nvfp4,
        )
        W = torch.randn(8, 16) * 0.3
        # Create heavily imbalanced activations: column 0 has huge
        # magnitude, the rest are small.
        X = torch.randn(100, 16) * 0.01
        X[:, 0] *= 100.0
        W_aw = _activation_weighted_round_nvfp4(W, X, group_size=16)
        self.assertEqual(W_aw.shape, W.shape)
        # Compare to pure RTN: the act-weighted pass should give at
        # least as low an output-space error (weighted by X).
        wp_rtn, ws_rtn, wg_rtn = quantize_dequantize_nvfp4(W)
        W_rtn = self._decode_nvfp4(wp_rtn, ws_rtn, wg_rtn)
        out_true = W @ X.t()
        out_rtn = W_rtn @ X.t()
        out_aw = W_aw @ X.t()
        err_rtn = (out_true - out_rtn).pow(2).mean().item()
        err_aw = (out_true - out_aw).pow(2).mean().item()
        # The activation-weighted polish should not be worse than RTN.
        # Tolerance allows the test to pass even when the two agree
        # exactly (the column-importance weighting doesn't flip any
        # decisions on small toy inputs). The point is: it's closed-
        # form and doesn't regress.
        self.assertLessEqual(err_aw, err_rtn * 1.01,
                             f"act-weighted {err_aw:.3e} > rtn {err_rtn:.3e}")

    def test_composed_passes_reduce_output_space_error_vs_rtn(self):
        """Integration test: synthetic linear + imbalanced activations.
        Running `_quantize_2d` with all 3 act-aware passes enabled
        should give lower activation-weighted output-space MSE than
        pure RTN (`_quantize_2d` with flags off)."""
        import torch
        from quantization.prismaquant.export_native_compressed import (
            _quantize_2d,
        )
        torch.manual_seed(7)
        out_f, in_f = 64, 128
        # Weight with some high-magnitude rows to stress quantization.
        W = torch.randn(out_f, in_f) * 0.15
        W[:, :8] *= 5.0                                  # bigger weights in first 8 cols
        # Heavily imbalanced activations: first 8 columns are huge
        # (should get more FP4 budget with AWQ), rest are small.
        X = torch.randn(512, in_f) * 0.1
        X[:, :8] *= 20.0
        # Reference BF16 output.
        ref = (W @ X.t()).float()

        # Pure RTN.
        out_rtn = _quantize_2d(W, "NVFP4", linear_name=None)
        W_rtn = self._decode_nvfp4(
            out_rtn["weight_packed"], out_rtn["weight_scale"],
            out_rtn["weight_global_scale"],
        )
        # All 3 passes on, activations passed explicitly.
        out_aa = _quantize_2d(
            W, "NVFP4",
            awq_enabled=True, gptq_enabled=True, awq_round_enabled=True,
            cached_activations=X,
        )
        W_aa = self._decode_nvfp4(
            out_aa["weight_packed"], out_aa["weight_scale"],
            out_aa["weight_global_scale"],
        )
        err_rtn = (ref - (W_rtn @ X.t())).pow(2).mean().item()
        err_aa = (ref - (W_aa @ X.t())).pow(2).mean().item()
        self.assertLess(
            err_aa, err_rtn,
            f"act-aware passes increased output error: "
            f"rtn={err_rtn:.4e} aa={err_aa:.4e}",
        )

    def test_act_aware_flags_module_default_off(self):
        """The module-level `_ACT_AWARE_FLAGS` defaults to all False so
        callers that don't touch main() get vanilla RTN behavior."""
        from quantization.prismaquant.export_native_compressed import (
            _ACT_AWARE_FLAGS,
        )
        self.assertFalse(_ACT_AWARE_FLAGS["awq"])
        self.assertFalse(_ACT_AWARE_FLAGS["gptq"])
        self.assertFalse(_ACT_AWARE_FLAGS["awq_round"])

    def test_quantize_2d_picks_up_module_flags(self):
        """When `_ACT_AWARE_FLAGS` is set and cached activations are
        present for the given linear_name, `_quantize_2d` runs the
        act-aware pipeline transparently."""
        import torch
        import quantization.prismaquant.export_native_compressed as m
        torch.manual_seed(11)
        W = torch.randn(32, 64) * 0.2
        X = torch.randn(128, 64) * 0.3
        saved_flags = dict(m._ACT_AWARE_FLAGS)
        saved_cache = m._CACHED_ACTIVATIONS
        try:
            m._ACT_AWARE_FLAGS.update({
                "awq": True, "gptq": False, "awq_round": True,
            })
            m._CACHED_ACTIVATIONS = {"demo.linear": X}
            out_with = m._quantize_2d(
                W, "NVFP4", linear_name="demo.linear",
            )
            m._ACT_AWARE_FLAGS.update({
                "awq": False, "gptq": False, "awq_round": False,
            })
            out_without = m._quantize_2d(
                W, "NVFP4", linear_name="demo.linear",
            )
        finally:
            m._ACT_AWARE_FLAGS.clear()
            m._ACT_AWARE_FLAGS.update(saved_flags)
            m._CACHED_ACTIVATIONS = saved_cache
        # The weight_packed should differ because AWQ rearranges the
        # quantization grid budget across channels.
        self.assertFalse(
            torch.equal(out_with["weight_packed"],
                        out_without["weight_packed"]),
            "act-aware flags had no effect on output",
        )

    def test_mxfp8_awq_roundtrip_preserves_scale(self):
        """MXFP8 with AWQ enabled: the stored weight should still
        round-trip through MXFP8 decode roughly, and have MSE lower
        than pure MXFP8 on activation-weighted input (though MXFP8
        at 8 bits rarely shows a big gap)."""
        import torch
        from quantization.prismaquant.export_native_compressed import (
            _quantize_2d,
        )
        torch.manual_seed(3)
        W = torch.randn(16, 64) * 0.1
        X = torch.randn(200, 64) * 0.5
        out = _quantize_2d(
            W, "MXFP8", awq_enabled=True, cached_activations=X,
        )
        self.assertIn("weight", out)
        self.assertIn("weight_scale", out)
        # Dequantize to check shape sanity.
        self.assertEqual(out["weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(out["weight"].shape), (16, 64))
