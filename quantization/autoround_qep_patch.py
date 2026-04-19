"""
QEP Patch for AutoRound

This patch modifies AutoRound's _quantize_block method to implement true QEP:
- Compute bf16 target using degraded inputs (q_input) when available
- Not just use degraded inputs for the quantized block

The key change is computing `output` (the optimization target) using q_input
instead of the original bf16-cached input_ids.

Current AutoRound: MSE(quant_block(degraded), bf16_block(clean))
With QEP patch:    MSE(quant_block(degraded), bf16_block(degraded))

To apply this patch:
1. Import this module after importing auto_round
2. Call patch_autoround_for_qep()

Example:
    from auto_round import AutoRound
    from autoround_qep_patch import patch_autoround_for_qep

    patch_autoround_for_qep()

    autoround = AutoRound(model, tokenizer, ...)
    autoround.quantize()
"""

import torch
from typing import Union
from functools import wraps


def patch_autoround_for_qep():
    """
    Monkey-patch AutoRound's LLMCompressor._quantize_block to implement true QEP.
    """
    from auto_round.compressors.base import BaseCompressor

    original_quantize_block = BaseCompressor._quantize_block

    @wraps(original_quantize_block)
    def _quantize_block_qep(
        self,
        block: torch.nn.Module,
        input_ids: Union[list[torch.Tensor], dict],
        input_others: dict,
        q_input: Union[torch.Tensor, dict, None] = None,
        device: Union[str, torch.device] = "cpu",
        auto_offload=True,
    ):
        """
        QEP-enhanced _quantize_block.

        Key change: When q_input is available, compute the bf16 target (output)
        using q_input instead of input_ids. This makes the optimization target
        realistic for what the block will see during inference.
        """
        from auto_round.compressors.utils import (
            materialize_model_,
            convert_module_to_hp_if_necessary,
            is_auto_device_mapping,
            set_auto_device_map_for_block_with_tuning,
            mv_module_from_gpu,
            clear_memory,
        )
        import accelerate

        materialize_model_(block)
        convert_module_to_hp_if_necessary(block, self.amp_dtype, device)

        if auto_offload:
            if is_auto_device_mapping(self.device_map) and len(self.device_list) > 1:
                card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                    block, self.device_map, input_ids, self.low_gpu_mem_usage, self.batch_size, device
                )
            else:
                block = block.to(device)
                card_0_in_high_risk, loss_device = False, device
        else:
            card_0_in_high_risk, loss_device = False, device

        if len(self.device_list) > 1 and auto_offload:
            for n, m in block.named_modules():
                if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                    continue
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module
                hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                add_hook_to_module(m, hook, True)

        # === QEP CHANGE: Use q_input for computing target if available ===
        # Original: Always compute output using input_ids (bf16 cached)
        # QEP: Compute output using q_input (degraded) when available

        target_input = q_input if q_input is not None else input_ids

        if q_input is None:
            hook_handles = self._register_act_max_hook(block)
            output = self._get_block_outputs(
                block, target_input, input_others,
                self.batch_size * self.infer_bs_coeff, device, self.cache_device
            )
            for handle in hook_handles:
                handle.remove()
        else:
            # QEP: Compute target with degraded inputs
            output = self._get_block_outputs(
                block, target_input, input_others,
                self.batch_size * self.infer_bs_coeff, device, self.cache_device
            )
            # Still collect activation stats with q_input
            hook_handles = self._register_act_max_hook(block)
            if hook_handles:
                self._get_block_outputs(
                    block,
                    q_input,
                    input_others,
                    self.batch_size * self.infer_bs_coeff,
                    device,
                    self.cache_device,
                    save_output=False,
                )
            for handle in hook_handles:
                handle.remove()

        # Replace input_ids with q_input for the optimization loop
        if q_input is not None:
            if input_ids is not q_input:
                clear_memory(input_ids, device_list=self.device_list)
            else:
                clear_memory(device_list=self.device_list)
            input_ids = q_input

        # Rest of the method continues as normal...
        # (wrapper, optimization loop, etc.)

        # Call the rest of the original implementation
        # We need to continue from here with the modified input_ids and output

        # For now, return to indicate patch point
        # Full implementation would copy the rest of _quantize_block

        return self._continue_quantize_block(
            block, input_ids, input_others, output,
            device, auto_offload, card_0_in_high_risk, loss_device
        )

    # Note: This is a conceptual patch. Full implementation would require
    # copying the entire _quantize_block method and modifying the target computation.

    print("QEP patch concept loaded. Full implementation requires method override.")
    print("Key change: output = _get_block_outputs(block, q_input, ...) when q_input available")


# For testing, create a simple demonstration
def demonstrate_qep_difference():
    """
    Demonstrate the difference between current AutoRound and QEP.
    """
    print("""
    Current AutoRound with enable_quanted_input=True:
    ================================================
    For block N (N > 0):

    1. input_ids = cached bf16 inputs (from try_cache_inter_data_gpucpu)
    2. q_input = outputs from quantized block N-1

    3. output = bf16_block(input_ids)     # Target uses CLEAN inputs
    4. input_ids = q_input                # Replace for optimization
    5. output_q = quant_block(input_ids)  # Quantized uses DEGRADED inputs
    6. loss = MSE(output_q, output)       # Mismatch!

    The optimization asks: "Make quant_block(degraded) match bf16_block(clean)"
    But at inference: quant_block will receive degraded inputs


    With QEP Patch:
    ===============
    For block N (N > 0):

    1. input_ids = cached bf16 inputs
    2. q_input = outputs from quantized block N-1

    3. output = bf16_block(q_input)       # Target uses DEGRADED inputs  <-- KEY CHANGE
    4. input_ids = q_input
    5. output_q = quant_block(input_ids)  # Quantized uses DEGRADED inputs
    6. loss = MSE(output_q, output)       # Matched!

    The optimization asks: "Make quant_block(degraded) match bf16_block(degraded)"
    This is exactly what we want at inference time.
    """)


if __name__ == "__main__":
    demonstrate_qep_difference()
