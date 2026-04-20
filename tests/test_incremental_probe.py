import json
import pickle
import re
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from quantization.prismquant.incremental_probe import (
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


if __name__ == "__main__":
    unittest.main()
