# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed-tensors scheme for pre-quantized MXFP8 weights.

Supports both MXFP8 (W8A8) and MXFP8A16 (W8A16) checkpoints serialized
with the ``mxfp8-quantized`` or ``float-quantized`` compressed-tensors
format.  Weights are stored as ``float8_e4m3fn`` with per-group (group_size=32)
``uint8`` E8M0 scales -- the standard MX block-scaling layout.

At inference time the existing ``Mxfp8LinearOp`` is reused, which selects
FlashInfer CUTLASS on SM100+ (Blackwell) or Marlin on SM80+ (Ampere/Ada).
"""

from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
    Mxfp8LinearOp,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

__all__ = ["CompressedTensorsMxfp8"]


class CompressedTensorsMxfp8(CompressedTensorsScheme):
    """
    Compressed tensors scheme for MXFP8 quantization (pre-quantized
    checkpoints).

    MXFP8 format:
    - 8-bit float weights (E4M3) stored as float8_e4m3fn
    - Per-group E8M0 scales (uint8) with group_size=32
    - Activations either dynamically quantized to MXFP8 (W8A8) or
      kept in BF16/FP16 (W8A16) -- the kernel handles both.
    """

    def __init__(self):
        self.group_size = MXFP8_BLOCK_SIZE  # 32
        self.mxfp8_linear = Mxfp8LinearOp()

    @classmethod
    def get_min_capability(cls) -> int:
        # Marlin supports MXFP8 on SM80+, FlashInfer CUTLASS on SM100+
        return 80

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
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        # FP8 E4M3 weights -- one byte per element (no packing)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Per-group E8M0 scales (uint8), one per block of 32 elements
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Convert from ModelWeightParameter to plain Parameter
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data, requires_grad=False
        )

        # Let Mxfp8LinearOp repack weights for the selected backend
        # (swizzle scales for FlashInfer CUTLASS, repack for Marlin, etc.)
        self.mxfp8_linear.process_weights(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mxfp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=layer.params_dtype,
            bias=bias,
            workspace=getattr(layer, "workspace", None),
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
        )
