import unittest

import torch
import torch.nn as nn

from quantization.prismquant.mtp_module import _load_into_mtp


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


class TestMtpLoadIntoPackedExperts(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
