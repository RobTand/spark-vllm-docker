import unittest

import torch

from quantization.dynaquant import format_registry as fr
from quantization.dynaquant.allocator import build_candidates
from quantization.dynaquant.calibrate_allocator import install_activation_hooks, select_targets


class TestDynaQuantFormatRegistry(unittest.TestCase):
    def test_block_formats_have_expected_shape_aware_bits(self):
        shape = (128, 128)
        self.assertAlmostEqual(fr.get_format("NVFP4").effective_bits_for_shape(shape), 4.5)
        self.assertAlmostEqual(fr.get_format("MXFP4").effective_bits_for_shape(shape), 4.25)
        self.assertAlmostEqual(fr.get_format("MXFP8").effective_bits_for_shape(shape), 8.25)
        self.assertAlmostEqual(fr.get_format("BF16").effective_bits_for_shape(shape), 16.0)

    def test_per_channel_formats_use_row_scale_count(self):
        shape = (5, 7)
        spec = fr.get_format("FP8_E4M3")
        expected_bytes = 5 * 7 + 5 * 4  # one byte per weight, fp32 row scales
        self.assertEqual(spec.scale_count_for_shape(shape), 5)
        self.assertEqual(spec.memory_bytes_for_shape(shape), expected_bytes)
        self.assertAlmostEqual(spec.effective_bits_for_shape(shape), 8.0 * expected_bytes / 35.0)

    def test_activation_quantization_changes_native_a4_a8_formats(self):
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        for fmt in ("NVFP4", "MXFP8", "FP8_E4M3"):
            y = fr.get_format(fmt).activation_quantize_dequantize(x.clone())
            self.assertFalse(torch.equal(x, y), msg=fmt)

    def test_activation_quantization_skips_a16_formats(self):
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        for fmt in ("MXFP8A16", "NVFP4A16", "BF16"):
            y = fr.get_format(fmt).activation_quantize_dequantize(x.clone())
            self.assertTrue(torch.equal(x, y), msg=fmt)


class TestDynaQuantAllocatorMath(unittest.TestCase):
    def test_build_candidates_uses_shape_aware_bits(self):
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 5,
                "in_features": 7,
                "n_params": 35,
            }
        }
        costs = {
            "layer.weight": {
                "FP8_E4M3": {"output_mse": 0.25},
                "BF16": {"output_mse": 0.0},
            }
        }
        cands = build_candidates(stats, costs, [fr.get_format("FP8_E4M3"), fr.get_format("BF16")])
        by_fmt = {cand.fmt: cand for cand in cands["layer.weight"]}
        self.assertAlmostEqual(
            by_fmt["FP8_E4M3"].bits_per_param,
            fr.get_format("FP8_E4M3").effective_bits_for_shape((5, 7)),
        )
        self.assertEqual(
            by_fmt["FP8_E4M3"].memory_bytes,
            fr.get_format("FP8_E4M3").memory_bytes_for_shape((5, 7)),
        )
        self.assertAlmostEqual(by_fmt["FP8_E4M3"].predicted_dloss, 0.5 * 2.0 * 0.25 * 5)

    def test_select_targets_returns_baseline_knee_high(self):
        curve = [
            {"feasible": True, "achieved_bits": 4.5, "predicted_dloss": 10.0},
            {"feasible": True, "achieved_bits": 5.0, "predicted_dloss": 4.0},
            {"feasible": True, "achieved_bits": 6.0, "predicted_dloss": 3.0},
            {"feasible": True, "achieved_bits": 8.0, "predicted_dloss": 2.8},
        ]
        picks = select_targets(curve, "baseline,knee,high")
        self.assertEqual(picks[0], 0)
        self.assertEqual(picks[-1], 3)
        self.assertEqual(len(picks), 3)


class TestCalibrationHooks(unittest.TestCase):
    def test_install_activation_hooks_skips_conflicting_module_formats(self):
        linear = torch.nn.Linear(4, 4, bias=False)
        quant_map = {
            "a.weight": (linear, "weight"),
            "b.weight": (linear, "weight"),
        }
        handles, active, skipped = install_activation_hooks(
            {"a.weight": "NVFP4", "b.weight": "MXFP8"},
            quant_map,
        )
        try:
            self.assertEqual(active, [])
            self.assertEqual(len(skipped), 1)
            self.assertEqual(set(skipped[0]["formats"]), {"MXFP8", "NVFP4"})
        finally:
            for handle in handles:
                handle.remove()

    def test_install_activation_hooks_quantizes_input(self):
        linear = torch.nn.Linear(4, 4, bias=False)
        quant_map = {"a.weight": (linear, "weight")}
        handles, active, skipped = install_activation_hooks({"a.weight": "NVFP4"}, quant_map)
        seen = {}

        def recorder(_mod, args):
            seen["input"] = args[0].detach().clone()

        capture = linear.register_forward_pre_hook(recorder)
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        try:
            linear(x)
            self.assertEqual(len(active), 1)
            self.assertEqual(skipped, [])
            self.assertFalse(torch.equal(seen["input"], x))
        finally:
            capture.remove()
            for handle in handles:
                handle.remove()


if __name__ == "__main__":
    unittest.main()
