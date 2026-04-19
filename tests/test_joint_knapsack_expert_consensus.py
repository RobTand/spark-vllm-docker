import unittest

from quantization.joint_knapsack_optimizer import (
    Config,
    ConfigResult,
    LayerInfo,
    aggregate_expert_families_for_optimization,
    apply_expert_consensus_to_recipe,
    expand_recipe_from_aggregates,
    recipe_cost_error,
)


def _layer(name, sensitivity, cfgs):
    return LayerInfo(
        name=name,
        shape=(2, 2),
        n_elements=4,
        sensitivity=sensitivity,
        pareto_configs=[
            ConfigResult(
                config=Config(*cfg),
                mse=mse,
                memory_bytes=mem,
                bits_per_weight=bpw,
            )
            for cfg, mse, mem, bpw in cfgs
        ],
        stats={"std": 0.0, "kurtosis": 0.0, "outlier_ratio": 0.0, "max_abs": 0.0},
    )


class TestExpertConsensus(unittest.TestCase):
    def test_aggregate_expert_families_for_optimization(self):
        common_cfgs = [
            ((4, 8, 16), 1.0, 10, 4.5),
            ((8, 16, 16), 0.2, 18, 9.0),
        ]
        lookup_layers = [
            _layer("model.layers.0.mlp.experts.gate_up_proj", 1.0, common_cfgs),
            _layer("model.layers.1.mlp.experts.gate_up_proj", 2.0, common_cfgs),
            _layer("model.layers.0.mlp.experts.down_proj", 1.5, common_cfgs),
            _layer("model.layers.1.mlp.experts.down_proj", 0.5, common_cfgs),
            _layer("model.layers.0.self_attn.q_proj.weight", 3.0, common_cfgs),
        ]

        optimized, aggregate_map = aggregate_expert_families_for_optimization(lookup_layers)
        names = {layer.name for layer in optimized}
        self.assertIn("moe_gate_up", names)
        self.assertIn("moe_down", names)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", names)
        self.assertEqual(aggregate_map["moe_gate_up"], [
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.1.mlp.experts.gate_up_proj",
        ])
        gate_up = next(layer for layer in optimized if layer.name == "moe_gate_up")
        self.assertEqual(gate_up.sensitivity, 1.0)
        self.assertEqual(gate_up.n_elements, 8)
        self.assertEqual(gate_up.pareto_configs[0].memory_bytes, 20)
        self.assertAlmostEqual(gate_up.pareto_configs[0].mse, 3.0)

        expanded = expand_recipe_from_aggregates(
            {
                "moe_gate_up": "w8_s16_g16",
                "moe_down": "w4_s8_g16",
                "model.layers.0.self_attn.q_proj.weight": "w4_s8_g16",
            },
            aggregate_map,
        )
        self.assertEqual(expanded["model.layers.0.mlp.experts.gate_up_proj"], "w8_s16_g16")
        self.assertEqual(expanded["model.layers.1.mlp.experts.gate_up_proj"], "w8_s16_g16")
        self.assertEqual(expanded["model.layers.0.mlp.experts.down_proj"], "w4_s8_g16")

    def test_consensus_collapses_expert_families(self):
        common_cfgs = [
            ((4, 8, 16), 1.0, 10, 4.5),
            ((5, 8, 16), 0.8, 12, 5.5),
            ((8, 16, 16), 0.2, 18, 9.0),
        ]
        lookup = {
            "model.layers.0.mlp.experts.gate_up_proj": _layer(
                "model.layers.0.mlp.experts.gate_up_proj", 1.0, common_cfgs
            ),
            "model.layers.1.mlp.experts.gate_up_proj": _layer(
                "model.layers.1.mlp.experts.gate_up_proj", 2.0, common_cfgs
            ),
            "model.layers.0.mlp.experts.down_proj": _layer(
                "model.layers.0.mlp.experts.down_proj", 1.0, common_cfgs
            ),
            "model.layers.1.mlp.experts.down_proj": _layer(
                "model.layers.1.mlp.experts.down_proj", 1.5, common_cfgs
            ),
            "model.layers.0.self_attn.q_proj.weight": _layer(
                "model.layers.0.self_attn.q_proj.weight", 1.0, common_cfgs
            ),
        }
        recipe = {
            "model.layers.0.mlp.experts.gate_up_proj": "w4_s8_g16",
            "model.layers.1.mlp.experts.gate_up_proj": "w8_s16_g16",
            "model.layers.0.mlp.experts.down_proj": "w5_s8_g16",
            "model.layers.1.mlp.experts.down_proj": "w8_s16_g16",
            "model.layers.0.self_attn.q_proj.weight": "w5_s8_g16",
        }

        adjusted, chosen = apply_expert_consensus_to_recipe(recipe, lookup)

        self.assertEqual(chosen["moe_gate_up"], "w5_s8_g16")
        self.assertEqual(chosen["moe_down"], "w8_s16_g16")
        self.assertEqual(adjusted["model.layers.0.mlp.experts.gate_up_proj"], "w5_s8_g16")
        self.assertEqual(adjusted["model.layers.1.mlp.experts.gate_up_proj"], "w5_s8_g16")
        self.assertEqual(adjusted["model.layers.0.mlp.experts.down_proj"], "w8_s16_g16")
        self.assertEqual(adjusted["model.layers.1.mlp.experts.down_proj"], "w8_s16_g16")
        self.assertEqual(adjusted["model.layers.0.self_attn.q_proj.weight"], "w5_s8_g16")

        cost, err, bpw = recipe_cost_error(adjusted, lookup)
        self.assertGreater(cost, 0)
        self.assertGreater(err, 0.0)
        self.assertGreater(bpw, 0.0)


if __name__ == "__main__":
    unittest.main()
