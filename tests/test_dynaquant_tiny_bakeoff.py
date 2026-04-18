import tempfile
import unittest
from types import SimpleNamespace

from quantization.dynaquant.tiny_bakeoff import build_bakeoff_commands


class TestTinyBakeoff(unittest.TestCase):
    def _args(self, skip_oracle=False):
        td = tempfile.mkdtemp()
        return SimpleNamespace(
            model="/tmp/model",
            probe="/tmp/probe.pkl",
            costs="/tmp/costs.pkl",
            activation_cache_dir="/tmp/act",
            formats="NVFP4,MXFP8,BF16",
            target_bits=4.8,
            top_units=6,
            neighbor_radius=1,
            n_calib_samples=2,
            calib_seqlen=64,
            device="cuda",
            oracle_max_combos=1024,
            output_dir=td,
            skip_oracle=skip_oracle,
            dry_run=True,
        )

    def test_build_bakeoff_commands_with_oracle(self):
        paths, cmds = build_bakeoff_commands(self._args(skip_oracle=False))
        self.assertEqual(len(cmds), 6)
        self.assertIn("quantization.dynaquant.calibrate_allocator", cmds[2])
        self.assertIn("quantization.dynaquant.quadratic_refine_allocator", cmds[3])
        self.assertIn("--calibration", cmds[3])
        self.assertIn(str(paths["calibration"]), cmds[3])
        self.assertIn("quantization.dynaquant.oracle_search", cmds[4])
        self.assertIn("--oracle", cmds[-1])
        self.assertTrue(str(paths["oracle"]).endswith("oracle.json"))

    def test_build_bakeoff_commands_without_oracle(self):
        _paths, cmds = build_bakeoff_commands(self._args(skip_oracle=True))
        self.assertEqual(len(cmds), 5)
        self.assertNotIn("quantization.dynaquant.oracle_search", " ".join(" ".join(c) for c in cmds))
        self.assertNotIn("--oracle", cmds[-1])


if __name__ == "__main__":
    unittest.main()
