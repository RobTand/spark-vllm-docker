"""
dynaquant_moe.py — DynaQuant FusedMoE method for vLLM.

Supports 1-16 bit per-expert quantization.  Each expert can have an
independently chosen bit width.  At inference, only the top-k active
experts are dequantized via fused Triton GEMV/GEMM kernels.

Memory-efficient: expert weights are loaded into a temporary dict during
weight loading, then compacted into flat buffers with exact sizing in
process_weights_after_loading.  No over-allocation.
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

_fused_linear = None


def _get_fused_linear():
    global _fused_linear
    if _fused_linear is None:
        from dynaquant.kernels.fused_dequant_gemv import dynaquant_linear
        _fused_linear = dynaquant_linear
    return _fused_linear


class DynaQuantFusedMoEMethod(FusedMoEMethodBase):
    """FusedMoE with per-expert DynaQuant (1-16 bit) quantization."""

    _logged = False

    def __init__(self, moe: FusedMoEConfig, group_size: int = 16,
                 max_bits: int = 8):
        super().__init__(moe)
        self.group_size = group_size
        self.max_bits = max_bits

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.dynaquant_num_experts = num_experts
        layer.dynaquant_hidden_size = hidden_size
        layer.dynaquant_intermediate_size = intermediate_size_per_partition
        layer.dynaquant_group_size = self.group_size

        inter = intermediate_size_per_partition
        gs = self.group_size
        n_groups_h = hidden_size // gs
        n_groups_i = inter // gs

        # ── Temporary storage: loaded data goes here first ──
        # Compacted into flat buffers in process_weights_after_loading.
        # This avoids over-allocation — only the actual loaded bytes exist.
        layer._dq_tmp = {
            "w13_packed": {},   # expert_id → [gate_bytes, up_bytes]
            "w2_packed": {},    # expert_id → bytes
            "w13_bits": {},     # expert_id → [gate_bits, up_bits]
            "w2_bits": {},      # expert_id → bits
        }

        # ── Scales: 3D (num_experts, rows, n_groups) — exact size ──
        w13_scale = torch.nn.Parameter(
            torch.zeros(num_experts, 2 * inter, n_groups_h, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)

        w2_scale = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, n_groups_i, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)

        # ── Dummy parameters for packed/bits so weight loader finds them ──
        # These are tiny placeholders; real data goes into _dq_tmp.
        w13_packed = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.uint8), requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_packed)

        w2_packed = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.uint8), requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_packed)

        w13_bits = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.int8), requires_grad=False,
        )
        layer.register_parameter("w13_weight_bits", w13_bits)

        w2_bits = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.int8), requires_grad=False,
        )
        layer.register_parameter("w2_weight_bits", w2_bits)

        # ── Dummy params for fp8 scale_inv (passthrough from original model) ──
        # The model's load_weights maps weight_scale_inv → w13/w2_weight_scale_inv.
        # We don't use them but need the parameters to exist for loading.
        w13_scale_inv = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_scale_inv)
        w2_scale_inv = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_scale_inv)

        # ── Custom weight loaders that store into _dq_tmp ──
        def _w13_packed_loader(param, loaded_weight, name="",
                               shard_id=None, expert_id=None):
            if expert_id is None:
                return
            tmp = layer._dq_tmp["w13_packed"]
            if expert_id not in tmp:
                tmp[expert_id] = [None, None]
            data = loaded_weight.view(-1).clone()
            if shard_id == "w1":
                tmp[expert_id][0] = data
            elif shard_id == "w3":
                tmp[expert_id][1] = data

        def _w2_packed_loader(param, loaded_weight, name="",
                              shard_id=None, expert_id=None):
            if expert_id is None:
                return
            layer._dq_tmp["w2_packed"][expert_id] = loaded_weight.view(-1).clone()

        def _w13_scale_loader(param, loaded_weight, name="",
                              shard_id=None, expert_id=None):
            if expert_id is None:
                return
            rows = loaded_weight.shape[0]
            if shard_id == "w1":
                param.data[expert_id, :rows] = loaded_weight
            elif shard_id == "w3":
                param.data[expert_id, inter:inter + rows] = loaded_weight

        def _w2_scale_loader(param, loaded_weight, name="",
                             shard_id=None, expert_id=None):
            if expert_id is None:
                return
            param.data[expert_id] = loaded_weight

        def _w13_bits_loader(param, loaded_weight, name="",
                             shard_id=None, expert_id=None):
            if expert_id is None:
                return
            tmp = layer._dq_tmp["w13_bits"]
            if expert_id not in tmp:
                tmp[expert_id] = [4, 4]
            val = int(loaded_weight.view(-1)[0].item())
            if shard_id == "w1":
                tmp[expert_id][0] = val
            elif shard_id == "w3":
                tmp[expert_id][1] = val

        def _w2_bits_loader(param, loaded_weight, name="",
                            shard_id=None, expert_id=None):
            if expert_id is None:
                return
            layer._dq_tmp["w2_bits"][expert_id] = int(loaded_weight.view(-1)[0].item())

        def _noop_loader(param, loaded_weight, name="",
                         shard_id=None, expert_id=None):
            pass  # scale_inv not used by DynaQuant

        # Set attrs without weight_loader, then set custom loaders
        attrs = {k: v for k, v in extra_weight_attrs.items()
                 if k != "weight_loader"}
        attrs["is_transposed"] = False
        for pname in ["w13_weight_packed", "w2_weight_packed",
                      "w13_weight_scale", "w2_weight_scale",
                      "w13_weight_bits", "w2_weight_bits",
                      "w13_weight_scale_inv", "w2_weight_scale_inv"]:
            set_weight_attrs(getattr(layer, pname), attrs)

        w13_packed.weight_loader = _w13_packed_loader
        w2_packed.weight_loader = _w2_packed_loader
        w13_scale.weight_loader = _w13_scale_loader
        w2_scale.weight_loader = _w2_scale_loader
        w13_bits.weight_loader = _w13_bits_loader
        w2_bits.weight_loader = _w2_bits_loader
        w13_scale_inv.weight_loader = _noop_loader
        w2_scale_inv.weight_loader = _noop_loader

        if not DynaQuantFusedMoEMethod._logged:
            logger.info(
                "DynaQuant MoE: %d experts, group=%d, deferred allocation",
                num_experts, self.group_size,
            )
            DynaQuantFusedMoEMethod._logged = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Compact expert data into flat buffers with exact sizing."""
        num_experts = layer.dynaquant_num_experts
        hidden = layer.dynaquant_hidden_size
        inter = layer.dynaquant_intermediate_size
        tmp = layer._dq_tmp

        # ── Build flat w13 packed buffer ──
        w13_chunks = []
        w13_table = []
        w13_offset = 0

        for e in range(num_experts):
            pair = tmp["w13_packed"].get(e, [None, None])
            bits = tmp["w13_bits"].get(e, [4, 4])
            gate_data = pair[0] if pair[0] is not None else torch.zeros(1, dtype=torch.uint8)
            up_data = pair[1] if pair[1] is not None else torch.zeros(1, dtype=torch.uint8)

            g_sz = gate_data.numel()
            u_sz = up_data.numel()
            g_bits = bits[0] or 4
            u_bits = bits[1] or g_bits

            w13_table.append((w13_offset, g_sz, w13_offset + g_sz, u_sz,
                              g_bits, u_bits, inter, hidden))
            w13_chunks.append(gate_data)
            w13_chunks.append(up_data)
            # 4 byte pad for kernel OOB guard
            w13_chunks.append(torch.zeros(4, dtype=torch.uint8))
            w13_offset += g_sz + u_sz + 4

        w13_flat = torch.cat(w13_chunks)
        layer.w13_weight_packed = torch.nn.Parameter(w13_flat, requires_grad=False)

        # ── Build flat w2 packed buffer ──
        w2_chunks = []
        w2_table = []
        w2_offset = 0

        for e in range(num_experts):
            data = tmp["w2_packed"].get(e, torch.zeros(1, dtype=torch.uint8))
            bits = tmp["w2_bits"].get(e, 4) or 4
            sz = data.numel()

            w2_table.append((w2_offset, sz, bits, hidden, inter))
            w2_chunks.append(data)
            w2_chunks.append(torch.zeros(4, dtype=torch.uint8))
            w2_offset += sz + 4

        w2_flat = torch.cat(w2_chunks)
        layer.w2_weight_packed = torch.nn.Parameter(w2_flat, requires_grad=False)

        layer.dynaquant_w13_table = w13_table
        layer.dynaquant_w2_table = w2_table

        # Free temporary storage
        del layer._dq_tmp

        total_mb = (w13_flat.numel() + w2_flat.numel()) / 1e6
        logger.debug("DynaQuant MoE: compacted %d experts, %.1f MB packed",
                      num_experts, total_mb)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused_linear = _get_fused_linear()
        gs = layer.dynaquant_group_size
        hidden = layer.dynaquant_hidden_size

        num_tokens = x.shape[0]
        output = torch.zeros(num_tokens, hidden, dtype=x.dtype, device=x.device)
        x_f32 = x.float()

        w13_packed = layer.w13_weight_packed.data
        w2_packed = layer.w2_weight_packed.data
        w13_scale = layer.w13_weight_scale.data
        w2_scale = layer.w2_weight_scale.data

        w13_table = layer.dynaquant_w13_table
        w2_table = layer.dynaquant_w2_table

        for expert_idx in topk_ids.unique():
            eidx = expert_idx.item()

            mask = (topk_ids == expert_idx)
            token_mask = mask.any(dim=1)
            if not token_mask.any():
                continue

            active = token_mask.nonzero(as_tuple=True)[0]
            ew = (topk_weights * mask.float()).sum(dim=1)[active].unsqueeze(1)
            xa = x_f32[active]

            # Gate (w1)
            g_off, g_sz, u_off, u_sz, g_bits, u_bits, out_f, in_f = w13_table[eidx]
            gate = fused_linear(xa, w13_packed[g_off:g_off + g_sz + 4],
                                w13_scale[eidx, :out_f], g_bits, out_f, in_f,
                                group_size=gs)

            # Up (w3)
            up = fused_linear(xa, w13_packed[u_off:u_off + u_sz + 4],
                              w13_scale[eidx, out_f:2 * out_f], u_bits, out_f, in_f,
                              group_size=gs)

            # SiLU activation
            h = F.silu(gate) * up

            # Down (w2)
            d_off, d_sz, d_bits, d_out, d_in = w2_table[eidx]
            down = fused_linear(h.float(), w2_packed[d_off:d_off + d_sz + 4],
                                w2_scale[eidx, :d_out], d_bits, d_out, d_in,
                                group_size=gs)

            output[active] += ew * down.to(x.dtype)

        return output
