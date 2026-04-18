import unittest

import torch

from quantization.dynaquant import format_registry as fr
from quantization.dynaquant.local_reconstruct import (
    _refine_measurement,
    _summarize_weight_clip,
    _sym_clip,
)


class TestLocalReconstruct(unittest.TestCase):
    def test_sym_clip_accepts_rowwise_tensor_factor(self):
        x = torch.tensor([[2.0, -1.0], [4.0, -3.0]], dtype=torch.float32)
        factor = torch.tensor([[0.5], [1.0]], dtype=torch.float32)
        y = _sym_clip(x, factor)
        self.assertTrue(torch.allclose(y[0], torch.tensor([1.0, -1.0])))
        self.assertTrue(torch.allclose(y[1], x[1]))

    def test_rowwise_refine_produces_rowwise_summary(self):
        torch.manual_seed(0)
        W = torch.tensor(
            [
                [4.0, -4.0, 0.2, -0.1],
                [0.3, -0.2, 0.1, -0.1],
                [0.4, 0.5, -0.2, 0.2],
            ],
            dtype=torch.float32,
        )
        X = torch.randn(16, 4)
        entry = _refine_measurement(
            W,
            X,
            fr.get_format("NVFP4"),
            [1.0, 0.98, 0.95],
            [1.0, 0.98],
            rounds=1,
            rowwise_topk=2,
            rowwise_rounds=1,
        )
        self.assertIsNotNone(entry)
        summary = _summarize_weight_clip(entry["weight_clip"])
        self.assertEqual(summary["mode"], "rowwise")
        self.assertEqual(len(summary["values"]), W.shape[0])


if __name__ == "__main__":
    unittest.main()
