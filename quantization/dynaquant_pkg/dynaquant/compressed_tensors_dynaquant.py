"""
compressed_tensors_dynaquant.py — vLLM compressed-tensors scheme for DynaQuant
arbitrary bit-width quantization (1-16 bits per projection).

On-disk per projection (e.g. gate_proj, up_proj, in_proj_qkv, etc.):
    {prefix}.weight_packed  — uint8 1D packed N-bit codes
    {prefix}.weight_scale   — fp32 (out_features, n_groups)
    {prefix}.weight_bits    — int8 scalar (the bit width for this projection)

Fused layers (gate_up, qkv, in_proj_qkvz) may receive fewer shards than
output partitions (e.g. in_proj_qkv covers q+k+v as one tensor).  The scheme
infers n_rows per shard from packed byte count and bit width, dispatching one
kernel call per shard.

Forward: fused Triton dequant+GEMV (batch=1) or dequant+GEMM (batch>1)
"""
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    GroupQuantScaleParameter,
)

logger = init_logger(__name__)
__all__ = ["CompressedTensorsDynaQuant"]

_fused_linear = None


def _get_fused_linear():
    global _fused_linear
    if _fused_linear is None:
        from dynaquant.kernels.fused_dequant_gemv import dynaquant_linear
        _fused_linear = dynaquant_linear
    return _fused_linear


class CompressedTensorsDynaQuant(CompressedTensorsScheme):
    """Mixed-precision quantization with 1-16 bits per projection.

    Each linear projection can have an independently chosen bit width.
    Sub-projections within fused layers (gate+up, q+k+v, qkvz) are NOT
    snapped to the same bit width — the scheme dispatches separate
    Triton kernels per loaded shard and concatenates the results.
    """

    _logged = False

    def __init__(self, num_bits: int = 4, group_size: int = 16):
        self.num_bits = num_bits
        self.group_size = group_size

    @classmethod
    def get_min_capability(cls) -> int:
        return 75  # Turing+

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        n_groups = input_size_per_partition // self.group_size

        # ── Store geometry on the layer for apply_weights ──
        layer.dynaquant_group_size = self.group_size
        layer.dynaquant_out_features = output_size_per_partition
        layer.dynaquant_in_features = input_size_per_partition
        layer.dynaquant_partition_sizes = output_partition_sizes

        # ── Scales: standard vLLM parameter (2D, output_dim=0) ──
        weight_scale = GroupQuantScaleParameter(
            data=torch.zeros(
                output_size_per_partition, n_groups, dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # ── Mutable state shared between loaders ──
        # Bits and packed data arrive in separate tensors and may arrive
        # in DIFFERENT ORDER from the scales (alphabetical vs shard_id).
        # We record shard_id with each load so process_weights_after_loading
        # can reorder to match the scale layout.
        load_state = {
            "packed_offset": 0,
            "packed_shards": [],  # [(shard_id, byte_offset, byte_count)]
            "bits_shards": [],    # [(shard_id, bits_value)]
        }
        layer._dynaquant_load_state = load_state

        # ── Packed weights: sequential byte append ──
        max_packed_bytes = (
            output_size_per_partition * input_size_per_partition * 16 + 7
        ) // 8 + 4

        def _packed_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: int | str | None = None,
        ):
            flat = loaded_weight.view(-1)
            n = flat.numel()
            off = load_state["packed_offset"]
            param.data[off:off + n] = flat[:n]
            load_state["packed_shards"].append((loaded_shard_id, off, n))
            load_state["packed_offset"] = off + n

        weight_packed = BasevLLMParameter(
            data=torch.zeros(max_packed_bytes, dtype=torch.uint8),
            weight_loader=_packed_weight_loader,
        )
        layer.register_parameter("weight_packed", weight_packed)

        # ── Bit widths: one int8 scalar per shard ──
        n_partitions = len(output_partition_sizes)

        def _bits_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: int | str | None = None,
        ):
            bits_val = int(loaded_weight.view(-1)[0].item())
            load_state["bits_shards"].append((loaded_shard_id, bits_val))

        weight_bits = BasevLLMParameter(
            data=torch.zeros(n_partitions, dtype=torch.int8),
            weight_loader=_bits_weight_loader,
        )
        layer.register_parameter("weight_bits", weight_bits)

        if not CompressedTensorsDynaQuant._logged:
            logger.info(
                "DynaQuant: 1-16 bit mixed-precision, group_size=%d, "
                "fused Triton kernels",
                self.group_size,
            )
            CompressedTensorsDynaQuant._logged = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Build per-shard dispatch table and trim the packed buffer.

        Each loaded shard becomes one kernel dispatch in apply_weights.
        A shard may cover multiple output partitions (e.g. in_proj_qkv
        covers q+k+v sub-projections at a single bit width).
        """
        load_state = layer._dynaquant_load_state
        packed_shards = load_state["packed_shards"]   # [(shard_id, byte_off, byte_count)]
        bits_shards = load_state["bits_shards"]       # [(shard_id, bits)]
        in_features = layer.dynaquant_in_features

        # Build bits lookup by shard_id
        bits_by_shard = {}
        for sid, bval in bits_shards:
            bits_by_shard[sid] = bval

        # Map shard_ids to canonical integer order for scale alignment.
        # Scales use GroupQuantScaleParameter with output_dim=0, which places
        # shards in the order defined by the linear layer's shard_id mapping:
        #   QKV: "q"=0, "k"=1, "v"=2   MLP: 0=gate, 1=up
        # We must sort packed shards into this same order.
        qkv_order = {"q": 0, "k": 1, "v": 2}

        def shard_sort_key(entry):
            sid = entry[0]
            if isinstance(sid, str):
                return qkv_order.get(sid, ord(sid))
            if isinstance(sid, int):
                return sid
            return 0  # None or unknown

        sorted_packed = sorted(packed_shards, key=shard_sort_key)

        partitions = []
        scale_row_offset = 0

        for sid, byte_off, byte_count in sorted_packed:
            bits = bits_by_shard.get(sid, self.num_bits)
            if bits <= 0:
                bits = self.num_bits

            n_rows = (byte_count * 8) // (in_features * bits)
            partitions.append(
                (bits, byte_off, byte_count, scale_row_offset, n_rows)
            )
            scale_row_offset += n_rows

        layer.dynaquant_partitions = partitions

        # Trim packed buffer to actual used size (+ 4 byte OOB guard)
        total_packed = load_state["packed_offset"]
        actual = total_packed + 4
        if actual < layer.weight_packed.data.numel():
            layer.weight_packed = torch.nn.Parameter(
                layer.weight_packed.data[:actual].clone(),
                requires_grad=False,
            )

        # Clean up
        del layer._dynaquant_load_state

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused_linear = _get_fused_linear()

        in_f = layer.dynaquant_in_features
        group_size = layer.dynaquant_group_size
        partitions = layer.dynaquant_partitions

        input_dtype = x.dtype
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        if x.shape[-1] < in_f:
            x = F.pad(x, (0, in_f - x.shape[-1]))

        x_f32 = x.float()
        packed = layer.weight_packed
        if isinstance(packed, torch.nn.Parameter):
            packed = packed.data
        scales = layer.weight_scale
        if isinstance(scales, torch.nn.Parameter):
            scales = scales.data

        if len(partitions) == 1:
            # ── Fast path: single shard ──
            bits, p_off, p_size, r_off, n_rows = partitions[0]
            p_end = min(p_off + p_size + 4, packed.numel())
            y = fused_linear(
                x_f32,
                packed[p_off:p_end],
                scales[r_off:r_off + n_rows],
                bits, n_rows, in_f,
                bias=bias,
                group_size=group_size,
            )
        else:
            # ── Multi-shard: heterogeneous bit widths ──
            parts = []
            for bits, p_off, p_size, r_off, n_rows in partitions:
                p_end = min(p_off + p_size + 4, packed.numel())
                y_part = fused_linear(
                    x_f32,
                    packed[p_off:p_end],
                    scales[r_off:r_off + n_rows],
                    bits, n_rows, in_f,
                    bias=None,
                    group_size=group_size,
                )
                parts.append(y_part)
            y = torch.cat(parts, dim=-1)
            if bias is not None:
                y = y + bias

        y = y.to(input_dtype)
        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], -1)
        return y
