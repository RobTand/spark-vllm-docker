import unittest

from quantization.export_mixed_native import (
    build_config_groups,
    canonicalize_format,
    explicit_regex,
    promote_serving_units,
    resolve_assignment,
)


class TestExportMixedNative(unittest.TestCase):
    def test_canonicalize_native_and_legacy(self):
        self.assertEqual(canonicalize_format("nvfp4", snap_legacy=False, allow_mxfp4=False), "NVFP4")
        self.assertEqual(canonicalize_format("mxfp8_e4m3", snap_legacy=False, allow_mxfp4=False), "MXFP8")
        self.assertEqual(canonicalize_format("bf16", snap_legacy=False, allow_mxfp4=False), "BF16")
        self.assertEqual(
            canonicalize_format("w8_s8_g32", snap_legacy=True, allow_mxfp4=False),
            "MXFP8",
        )

    def test_promote_serving_units(self):
        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "MXFP8",
            "model.layers.0.self_attn.v_proj": "NVFP4",
            "model.layers.0.self_attn.o_proj": "NVFP4",
            "model.layers.0.mlp.gate_proj": "NVFP4",
            "model.layers.0.mlp.up_proj": "FP8",
        }
        promoted = promote_serving_units(assignment)
        self.assertEqual(promoted["model.layers.0.self_attn.q_proj"], "MXFP8")
        self.assertEqual(promoted["model.layers.0.self_attn.k_proj"], "MXFP8")
        self.assertEqual(promoted["model.layers.0.self_attn.v_proj"], "MXFP8")
        self.assertEqual(promoted["model.layers.0.self_attn.o_proj"], "NVFP4")
        self.assertEqual(promoted["model.layers.0.mlp.gate_proj"], "FP8")
        self.assertEqual(promoted["model.layers.0.mlp.up_proj"], "FP8")

    def test_resolve_assignment_strips_weight_suffix(self):
        raw = {
            "model.layers.0.self_attn.q_proj.weight": "NVFP4",
            "model.layers.0.self_attn.k_proj.weight": "MXFP8",
            "model.layers.0.self_attn.v_proj.weight": "NVFP4",
        }
        resolved = resolve_assignment(raw, snap_legacy=False, allow_mxfp4=False)
        self.assertIn("model.layers.0.self_attn.q_proj", resolved)
        self.assertNotIn("model.layers.0.self_attn.q_proj.weight", resolved)
        self.assertEqual(resolved["model.layers.0.self_attn.q_proj"], "MXFP8")

    def test_build_config_groups_uses_explicit_targets_and_bf16_ignore(self):
        assignment = {
            "model.layers.0.self_attn.q_proj": "MXFP8",
            "model.layers.0.self_attn.k_proj": "MXFP8",
            "model.layers.0.self_attn.v_proj": "MXFP8",
            "model.layers.0.self_attn.o_proj": "NVFP4",
            "model.layers.0.mlp.down_proj": "BF16",
        }
        groups, ignore, default_format = build_config_groups(
            assignment,
            ignore=["lm_head"],
            mxfp8_flavor="W8A8",
            fp8_flavor="dynamic",
        )
        # Two groups emitted: one explicit for the non-default format
        # (MXFP8 — q/k/v), one "Linear" catchall for the default
        # (NVFP4 — o_proj). BF16 + ignored live in `ignore`.
        # Default is the lowest-rank format present (NVFP4 ranks 0 per
        # FORMAT_RANK; MXFP8 ranks 2) — it gets the catchall target.
        self.assertEqual(set(groups.keys()), {"group_0", "group_1"})
        self.assertEqual(default_format, "NVFP4")
        self.assertIn("model.layers.0.mlp.down_proj", ignore)
        self.assertIn("lm_head", ignore)
        # Explicit MXFP8 targets land in the non-catchall group.
        explicit_targets = []
        for group in groups.values():
            if group["targets"] != ["Linear"]:
                explicit_targets.extend(group["targets"])
        self.assertIn(explicit_regex("model.layers.0.self_attn.q_proj"),
                      explicit_targets)
        self.assertIn(explicit_regex("model.layers.0.self_attn.v_proj"),
                      explicit_targets)
        # NVFP4 (the default) goes into the "Linear" catchall group —
        # o_proj gets picked up there, not via an explicit regex.
        all_targets = []
        for group in groups.values():
            all_targets.extend(group["targets"])
        self.assertIn("Linear", all_targets)
        self.assertNotIn(explicit_regex("model.layers.0.self_attn.o_proj"),
                         explicit_targets)


if __name__ == "__main__":
    unittest.main()
