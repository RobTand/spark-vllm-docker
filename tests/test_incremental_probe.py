import pickle
import tempfile
import unittest
from pathlib import Path

from quantization.dynaquant.incremental_probe import (
    build_layer_shard_regexes,
    merge_probe_pickles,
)


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
            pickle.dump({
                "stats": {"layer.0": {"h_trace": 1.0}},
                "router_counts": {"r": {"0": 1.5}},
                "router_totals": {"r": 3},
                "expert_info": {"layer.0": ("r", "0")},
                "meta": {"model": "toy"},
            }, open(p1, "wb"))
            pickle.dump({
                "stats": {"layer.1": {"h_trace": 2.0}},
                "router_counts": {"r": {"0": 0.5, "1": 2.0}},
                "router_totals": {"r": 5},
                "expert_info": {"layer.1": ("r", "1")},
                "meta": {"model": "toy"},
            }, open(p2, "wb"))

            merge_probe_pickles([p1, p2], out)
            merged = pickle.load(open(out, "rb"))
            self.assertEqual(set(merged["stats"]), {"layer.0", "layer.1"})
            self.assertEqual(merged["router_counts"]["r"]["0"], 2.0)
            self.assertEqual(merged["router_counts"]["r"]["1"], 2.0)
            self.assertEqual(merged["router_totals"]["r"], 8)
            self.assertEqual(merged["meta"]["n_shards"], 2)


if __name__ == "__main__":
    unittest.main()
