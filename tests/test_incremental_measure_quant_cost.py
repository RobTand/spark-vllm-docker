import pickle
import tempfile
import unittest
from pathlib import Path

from quantization.prismquant.incremental_measure_quant_cost import (
    cost_shard_is_reusable,
    merge_cost_pickles,
)


class TestIncrementalMeasureQuantCost(unittest.TestCase):
    def test_merge_cost_pickles_combines_disjoint_shards(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p1 = td / "a.pkl"
            p2 = td / "b.pkl"
            out = td / "merged.pkl"
            with open(p1, "wb") as f:
                pickle.dump({
                    "costs": {"layer.0": {"NVFP4": {"output_mse": 1.0}}},
                    "formats": ["NVFP4"],
                    "meta": {"part": 1},
                }, f)
            with open(p2, "wb") as f:
                pickle.dump({
                    "costs": {"layer.1": {"BF16": {"output_mse": 0.0}}},
                    "formats": ["NVFP4"],
                    "meta": {"part": 2},
                }, f)

            merge_cost_pickles([p1, p2], out)
            with open(out, "rb") as f:
                merged = pickle.load(f)
            self.assertEqual(set(merged["costs"]), {"layer.0", "layer.1"})
            self.assertEqual(merged["formats"], ["NVFP4"])
            self.assertEqual(merged["meta"]["n_shards"], 2)

    def test_cost_shard_reuse_requires_matching_incremental_meta(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shard = td / "cost_shard.pkl"
            with open(shard, "wb") as f:
                pickle.dump({
                    "costs": {"layer.0": {"NVFP4": {"output_mse": 1.0}}},
                    "formats": ["NVFP4", "MXFP8"],
                    "meta": {
                        "model": "toy-model",
                        "probe": str(td / "probe_subset_000.pkl"),
                        "mode": "auto",
                        "incremental_shard": {
                            "activation_cache_dir": str(td / "act"),
                            "linear_include": r"model\\.layers\\.0\\.",
                            "chunk_size": 256,
                            "h_detail_dir": None,
                            "shard_idx": 0,
                        },
                    },
                }, f)

            expected = {
                "model": "toy-model",
                "probe": str(td / "probe_subset_000.pkl"),
                "activation_cache_dir": str(td / "act"),
                "linear_include": r"model\\.layers\\.0\\.",
                "mode": "auto",
                "chunk_size": 256,
                "h_detail_dir": None,
                "shard_idx": 0,
                "formats": ["NVFP4", "MXFP8"],
            }
            self.assertTrue(cost_shard_is_reusable(shard, expected))

            stale = dict(expected)
            stale["formats"] = ["NVFP4"]
            self.assertFalse(cost_shard_is_reusable(shard, stale))


if __name__ == "__main__":
    unittest.main()
