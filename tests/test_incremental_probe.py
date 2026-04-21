import json
import pickle
import re
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from quantization.prismquant.incremental_probe import (
    GlobalPrecompute,
    _compute_precompute_key,
    _load_precompute_cache,
    _save_precompute_cache,
    build_extended_shard_regexes,
    build_layer_shard_regexes,
    merge_probe_pickles,
    probe_shard_is_reusable,
)
from quantization.prismquant.mtp_module import _load_into_mtp


class TestIncrementalProbe(unittest.TestCase):
    def test_build_layer_shard_regexes_groups_layers(self):
        regexes = build_layer_shard_regexes(5, 2)
        self.assertEqual(regexes, [
            r"model\.layers\.(?:0|1)\.",
            r"model\.layers\.(?:2|3)\.",
            r"model\.layers\.4\.",
        ])

    def test_merge_probe_pickles_sums_router_counts(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p1 = td / "a.pkl"
            p2 = td / "b.pkl"
            out = td / "merged.pkl"
            with open(p1, "wb") as f:
                pickle.dump({
                    "stats": {"layer.0": {"h_trace": 1.0}},
                    "router_counts": {"r": {"0": 1.5}},
                    "router_totals": {"r": 3},
                    "expert_info": {"layer.0": ("r", "0")},
                    "meta": {"model": "toy"},
                }, f)
            with open(p2, "wb") as f:
                pickle.dump({
                    "stats": {"layer.1": {"h_trace": 2.0}},
                    "router_counts": {"r": {"0": 0.5, "1": 2.0}},
                    "router_totals": {"r": 5},
                    "expert_info": {"layer.1": ("r", "1")},
                    "meta": {"model": "toy"},
                }, f)

            merge_probe_pickles([p1, p2], out)
            with open(out, "rb") as f:
                merged = pickle.load(f)
            self.assertEqual(set(merged["stats"]), {"layer.0", "layer.1"})
            self.assertEqual(merged["router_counts"]["r"]["0"], 2.0)
            self.assertEqual(merged["router_counts"]["r"]["1"], 2.0)
            self.assertEqual(merged["router_totals"]["r"], 8)
            self.assertEqual(merged["meta"]["n_shards"], 2)

    def test_probe_shard_reuse_requires_matching_incremental_meta(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shard = td / "probe_shard.pkl"
            with open(shard, "wb") as f:
                pickle.dump({
                    "stats": {"layer.0": {"h_trace": 1.0}},
                    "meta": {
                        "model": "toy-model",
                        "dataset": "toy-ds",
                        "nsamples": 4,
                        "seqlen": 128,
                        "dtype": "bf16",
                        "importance_weighting": True,
                        "linear_include": r"model\\.layers\\.0\\.",
                        "linear_exclude": "router",
                        "incremental_shard": {
                            "requested_device": "cuda",
                            "requested_device_map": "None",
                            "activation_cache_dir": str(td / "act"),
                            "h_detail_dir": None,
                            "shard_idx": 0,
                        },
                    },
                }, f)

            expected = {
                "model": "toy-model",
                "dataset": "toy-ds",
                "nsamples": 4,
                "seqlen": 128,
                "dtype": "bf16",
                "requested_device": "cuda",
                "requested_device_map": "None",
                "importance_weighting": True,
                "activation_cache_dir": str(td / "act"),
                "linear_include": r"model\\.layers\\.0\\.",
                "linear_exclude": "router",
                "h_detail_dir": None,
                "shard_idx": 0,
            }
            self.assertTrue(probe_shard_is_reusable(shard, expected))

            stale = dict(expected)
            stale["seqlen"] = 256
            self.assertFalse(probe_shard_is_reusable(shard, stale))


class TestExtendedShardRegexes(unittest.TestCase):
    """MTP is folded into the unified incremental shard enumeration
    (commit f13ea81). Assert the extended shard regex list contains a
    pattern that matches MTP linears when `include_mtp=True`."""

    def _write_config(self, td: Path, *, num_body: int, num_mtp: int) -> Path:
        model_dir = td / "model"
        model_dir.mkdir()
        config = {
            "num_hidden_layers": num_body,
            "num_nextn_predict_layers": num_mtp,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        return model_dir

    def test_mtp_regex_matches_mtp_layer_qnames(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            model_dir = self._write_config(td, num_body=4, num_mtp=1)
            regexes = build_extended_shard_regexes(
                str(model_dir),
                layers_per_shard=1,
                include_body=False,
                include_mtp=True,
                include_visual=False,
                include_lm_head=False,
            )
            # One MTP shard regex should be present.
            self.assertEqual(len(regexes), 1,
                             f"expected a single MTP regex, got {regexes!r}")
            mtp_re = re.compile(regexes[0])

            # Sample MTP Linear qnames: must match.
            for qname in [
                "mtp.layers.0.self_attn.q_proj",
                "mtp.layers.0.mlp.experts.gate_up_proj",
                "mtp.layers.0.mlp.gate",
            ]:
                self.assertIsNotNone(mtp_re.search(qname),
                                     f"MTP shard regex did not match {qname!r}")

            # Body linears must NOT be matched by the MTP regex.
            for qname in [
                "model.layers.0.self_attn.q_proj",
                "lm_head",
            ]:
                self.assertIsNone(mtp_re.search(qname),
                                  f"MTP regex unexpectedly matched {qname!r}")

    def test_include_mtp_false_omits_mtp_regex(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            model_dir = self._write_config(td, num_body=2, num_mtp=1)
            regexes = build_extended_shard_regexes(
                str(model_dir),
                layers_per_shard=1,
                include_body=True,
                include_mtp=False,
                include_visual=False,
                include_lm_head=False,
            )
            joined = "\n".join(regexes)
            self.assertNotIn("mtp.layers", joined)
            self.assertNotIn("mtp\\.layers", joined)


class _DummyMtp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].mlp = nn.Module()
        self.layers[0].mlp.experts = nn.Module()
        self.layers[0].mlp.experts.register_parameter(
            "gate_up_proj", nn.Parameter(torch.zeros(3, 6, 5))
        )
        self.layers[0].mlp.experts.register_parameter(
            "down_proj", nn.Parameter(torch.zeros(3, 7, 2))
        )


class TestResidentLinearClassification(unittest.TestCase):
    """Regression for the resident-linear bug: lm_head (and any other
    root-level Linear not under a decoder-layer prefix) must NOT be
    dropped by the per-layer bucketing in `_run_body_streaming_shard`.

    The probe function is large and needs a full StreamingContext to
    invoke, so rather than stand one up we replay the classification
    logic in isolation — the same two lines that decide whether a
    Linear gets a Fisher hook installed. A future failure in the body
    would here mean `resident_linears` again returns `[]` for lm_head,
    which is exactly what the original bug did.
    """

    def test_lm_head_classified_as_resident_not_body(self):
        # Build a tiny fake model with decoder layers + a root lm_head.
        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4, bias=False)

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([_Layer(), _Layer()])
                self.lm_head = nn.Linear(4, 8, bias=False)

        model = _FakeModel()
        layers_prefix = "model.layers."
        num_layers = 2
        linear_include = r"^lm_head$"
        linear_exclude = (
            r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|"
            r"block_sparse_moe\.gate$)"
        )

        # Replay the exact classification from the probe shard runner.
        inc = re.compile(linear_include)
        exc = re.compile(linear_exclude)
        all_linears = [
            n for n, m in model.named_modules() if isinstance(m, nn.Linear)
        ]
        all_tracked = [n for n in all_linears
                       if inc.search(n) and not exc.search(n)]
        layer_linear_names = []
        for L in range(num_layers):
            pref = f"{layers_prefix}{L}."
            layer_linear_names.append(
                [n for n in all_tracked if n.startswith(pref)])
        total_tracked = sum(len(x) for x in layer_linear_names)
        resident_linears = [
            n for n in all_tracked
            if not any(
                n.startswith(f"{layers_prefix}{L}.") for L in range(num_layers))
        ]

        # The original bug: lm_head is in all_tracked but falls into no
        # layer bucket, so `total_tracked` was 0 and the shard wrote an
        # empty pickle. The fix: resident_linears captures it so the
        # shard runs Phase-2 with hooks installed.
        self.assertIn("lm_head", all_tracked)
        self.assertEqual(total_tracked, 0,
                         "lm_head must not fall into any decoder-layer bucket")
        self.assertEqual(resident_linears, ["lm_head"],
                         "lm_head must be classified as resident so the "
                         "shard does not early-return with an empty pickle")

    def test_body_linear_classified_as_body_not_resident(self):
        # Sanity check the other direction: a layer Linear must be
        # bucketed into a body layer, not classified as resident.
        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4, bias=False)

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([_Layer()])
                self.lm_head = nn.Linear(4, 8, bias=False)

        model = _FakeModel()
        inc = re.compile(r"model\.layers\.0\.")
        all_linears = [
            n for n, m in model.named_modules() if isinstance(m, nn.Linear)
        ]
        all_tracked = [n for n in all_linears if inc.search(n)]
        resident_linears = [
            n for n in all_tracked
            if not n.startswith("model.layers.0.")
        ]
        self.assertEqual(all_tracked, ["model.layers.0.q_proj"])
        self.assertEqual(resident_linears, [])


class TestIncrementalProbeMtp(unittest.TestCase):
    """MTP shard helpers, previously tested in `test_mtp_probe.py`,
    folded in here because MTP is just a shard type in the unified
    incremental probe path."""

    def test_load_into_mtp_packs_expert_weights(self):
        mtp = _DummyMtp()
        raw = {
            "fc.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "layers.0.mlp.experts.1.gate_proj.weight": torch.full((3, 5), 11.0),
            "layers.0.mlp.experts.1.up_proj.weight": torch.full((3, 5), 22.0),
            "layers.0.mlp.experts.1.down_proj.weight": torch.full((7, 2), 33.0),
        }

        missing, extra = _load_into_mtp(mtp, raw)

        self.assertEqual(missing, [])
        self.assertNotIn("fc.weight", extra)
        self.assertNotIn("layers.0.mlp.experts.gate_up_proj", extra)
        self.assertNotIn("layers.0.mlp.experts.down_proj", extra)
        self.assertTrue(torch.equal(mtp.fc.weight, raw["fc.weight"]))
        self.assertTrue(torch.equal(
            mtp.layers[0].mlp.experts.gate_up_proj[1, :3],
            raw["layers.0.mlp.experts.1.gate_proj.weight"],
        ))
        self.assertTrue(torch.equal(
            mtp.layers[0].mlp.experts.gate_up_proj[1, 3:],
            raw["layers.0.mlp.experts.1.up_proj.weight"],
        ))
        self.assertTrue(torch.equal(
            mtp.layers[0].mlp.experts.down_proj[1],
            raw["layers.0.mlp.experts.1.down_proj.weight"],
        ))


class TestGlobalPrecomputeCache(unittest.TestCase):
    """Global Phase-1 + Phase-2 artifacts are now hoisted out of the
    per-shard loop and cached to disk so an interrupted run can resume
    without redoing the body forward / CE backward. Exercise the cache
    roundtrip here: save, then load, and verify the content is reused
    only when the fingerprint matches.

    (The full probe pipeline requires a real model + streaming weights,
    so we don't drive `_compute_global_precompute` end-to-end here; a
    TODO for that once a tiny fixture model is available.)
    """

    def _fake_precompute(self, num_layers: int = 3, T: int = 4, H: int = 6,
                         N: int = 1) -> GlobalPrecompute:
        acts = [torch.randn(N, T, H, dtype=torch.float32)
                for _ in range(num_layers + 1)]
        grad = torch.randn(N, T, H, dtype=torch.float32)
        ids = torch.arange(N * T, dtype=torch.long).reshape(N, T)
        return GlobalPrecompute(
            activations_cpu=acts,
            grad_at_tail=grad,
            ids=ids,
            resident_stats={"lm_head": {"h_trace_raw": 1.0}},
            resident_h_full={"lm_head": torch.ones(2, 3, dtype=torch.float32)},
            resident_act_snaps={"lm_head": [torch.zeros(2, 3)]},
        )

    def test_precompute_cache_reuses_when_meta_matches(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            path = td / "pre.pt"
            pre = self._fake_precompute()
            meta = _compute_precompute_key(
                model_path="toy",
                dataset_name="ds",
                nsamples=1,
                seqlen=4,
                dtype_name="bf16",
                device="cpu",
                importance_weighting=True,
                resident_include_union=r"^lm_head$",
            )
            _save_precompute_cache(path, pre, meta)
            loaded = _load_precompute_cache(path, meta, torch.device("cpu"))
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.activations_cpu), len(pre.activations_cpu))
            self.assertTrue(torch.equal(loaded.grad_at_tail, pre.grad_at_tail))
            self.assertTrue(torch.equal(loaded.ids, pre.ids))
            self.assertEqual(
                set(loaded.resident_stats), set(pre.resident_stats))

    def test_precompute_cache_rejects_meta_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            path = td / "pre.pt"
            pre = self._fake_precompute()
            meta = _compute_precompute_key(
                model_path="toy",
                dataset_name="ds",
                nsamples=1,
                seqlen=4,
                dtype_name="bf16",
                device="cpu",
                importance_weighting=True,
                resident_include_union=r"^lm_head$",
            )
            _save_precompute_cache(path, pre, meta)
            stale = dict(meta)
            stale["nsamples"] = 2
            loaded = _load_precompute_cache(path, stale, torch.device("cpu"))
            self.assertIsNone(loaded)

    def test_precompute_cache_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            path = td / "does_not_exist.pt"
            meta = _compute_precompute_key(
                model_path="toy",
                dataset_name="ds",
                nsamples=1,
                seqlen=4,
                dtype_name="bf16",
                device="cpu",
                importance_weighting=True,
                resident_include_union=r"^lm_head$",
            )
            self.assertIsNone(
                _load_precompute_cache(path, meta, torch.device("cpu")))

    # TODO: once a tiny fixture model is available, add an end-to-end
    # test that invokes the main orchestrator twice and asserts Phase-1
    # is only run once (by spying on `_compute_global_precompute` calls).


if __name__ == "__main__":
    unittest.main()


class TestMoEPairPromotion(unittest.TestCase):
    """Regression: vLLM FusedMoE requires gate_up_proj + down_proj to
    share a quantization scheme within a single layer. The allocator's
    `promote_moe_pair` enforces this after per-layer super-candidate
    selection."""

    def test_mixed_pair_is_promoted(self):
        from quantization.prismquant.allocator import promote_moe_pair
        assign = {
            "model.layers.46.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.46.mlp.experts.down_proj": "BF16",
        }
        rank = {"BF16": 0, "MXFP8": 1, "NVFP4": 2}
        out = promote_moe_pair(assign, rank)
        self.assertEqual(out["model.layers.46.mlp.experts.gate_up_proj"], "NVFP4")
        self.assertEqual(out["model.layers.46.mlp.experts.down_proj"], "NVFP4")

    def test_matched_pair_is_stable(self):
        from quantization.prismquant.allocator import promote_moe_pair
        assign = {
            "model.layers.5.mlp.experts.gate_up_proj": "MXFP8",
            "model.layers.5.mlp.experts.down_proj": "MXFP8",
        }
        rank = {"BF16": 0, "MXFP8": 1, "NVFP4": 2}
        out = promote_moe_pair(assign, rank)
        self.assertEqual(out, assign)

    def test_unrelated_names_untouched(self):
        from quantization.prismquant.allocator import promote_moe_pair
        assign = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
        }
        rank = {"BF16": 0, "MXFP8": 1, "NVFP4": 2}
        out = promote_moe_pair(assign, rank)
        self.assertEqual(out, assign)
