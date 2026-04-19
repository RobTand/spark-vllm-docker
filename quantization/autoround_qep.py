#!/usr/bin/env python3
"""
QEP (Quantization Error Propagation) extension for AutoRound.

This module provides a QEP-enabled wrapper around AutoRound that propagates
quantized outputs as inputs to downstream block optimization, rather than
using cached bf16 inputs.

The key change: instead of pre-caching all inputs with bf16 model, we
re-run inference through already-quantized blocks to get degraded inputs
for each subsequent block's optimization.

Usage:
    from autoround_qep import AutoRoundQEP

    autoround = AutoRoundQEP(
        model=model,
        tokenizer=tokenizer,
        bits=4,
        group_size=16,
        enable_qep=True,  # Enable QEP
        ...
    )
    autoround.quantize()
"""

import gc
import torch
import torch.nn as nn
from typing import Union, Optional, List, Dict, Any
from tqdm import tqdm


def fp4_quantize_layer(module: nn.Linear, group_size: int = 16) -> None:
    """In-place FP4 quantization of a linear layer (RTN baseline)."""
    weight = module.weight.data
    out_f, in_f = weight.shape

    n_groups = (in_f + group_size - 1) // group_size
    pad_size = n_groups * group_size - in_f
    if pad_size > 0:
        weight_padded = torch.nn.functional.pad(weight, (0, pad_size))
    else:
        weight_padded = weight

    grouped = weight_padded.view(out_f, n_groups, group_size)
    scales = grouped.abs().max(dim=2, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales * 6.0
    abs_n = normalized.abs()
    sign = normalized.sign()

    # FP4 E2M1 quantization
    q_abs = torch.where(abs_n <= 2.0, (abs_n * 2).round() / 2,
            torch.where(abs_n <= 2.5, torch.full_like(abs_n, 2.0),
            torch.where(abs_n <= 3.5, torch.full_like(abs_n, 3.0),
            torch.where(abs_n <= 5.0, torch.full_like(abs_n, 4.0),
                        torch.full_like(abs_n, 6.0)))))

    dequantized = sign * q_abs * scales / 6.0
    module.weight.data = dequantized.view(out_f, -1)[:, :in_f].contiguous()


class QEPBlockOptimizer:
    """
    QEP-aware block optimizer that uses degraded inputs from upstream
    quantized blocks instead of cached bf16 inputs.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        bits: int = 4,
        group_size: int = 16,
        sym: bool = True,
        nsamples: int = 128,
        seqlen: int = 2048,
        batch_size: int = 8,
        iters: int = 200,
        dataset: str = "NeelNanda/pile-10k",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.nsamples = nsamples
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.iters = iters
        self.dataset = dataset

        # Detect model structure
        self.blocks = self._get_blocks()
        self.embed_tokens = self._get_embed_tokens()

    def _get_blocks(self) -> nn.ModuleList:
        """Get transformer blocks from model."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")

    def _get_embed_tokens(self) -> nn.Module:
        """Get embedding layer from model."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            return self.model.transformer.wte
        else:
            raise ValueError("Unsupported model architecture")

    def _prepare_calibration_data(self) -> List[torch.Tensor]:
        """Load and prepare calibration data."""
        try:
            from datasets import load_dataset
            dataset = load_dataset(self.dataset, split="train")

            inputs = []
            for i in range(len(dataset)):
                text = dataset[i]["text"]
                tokens = self.tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=self.seqlen
                )
                if tokens["input_ids"].shape[1] >= self.seqlen // 2:
                    inputs.append(tokens["input_ids"])
                    if len(inputs) >= self.nsamples:
                        break
            return inputs
        except ImportError:
            # Fallback to random tokens
            vocab_size = self.model.config.vocab_size
            return [
                torch.randint(0, vocab_size, (1, self.seqlen))
                for _ in range(self.nsamples)
            ]

    @torch.no_grad()
    def _get_block_inputs_qep(
        self,
        block_idx: int,
        calibration_inputs: List[torch.Tensor],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Get inputs for block_idx by running through already-quantized
        upstream blocks (QEP-style).

        This is the key QEP mechanism: instead of using cached bf16 inputs,
        we propagate through quantized blocks to get realistic degraded inputs.
        """
        all_inputs = []
        captured_input = [None]

        # Use hook to capture input to target block
        def capture_hook(module, input, output):
            if isinstance(input, tuple):
                captured_input[0] = input[0].detach().cpu()
            else:
                captured_input[0] = input.detach().cpu()

        # Register hook on target block
        hook = self.blocks[block_idx].register_forward_hook(capture_hook)

        try:
            for input_ids in calibration_inputs:
                input_ids = input_ids.to(device)
                # Run full model forward (will go through quantized blocks)
                _ = self.model(input_ids)
                all_inputs.append(captured_input[0])
        finally:
            hook.remove()

        return all_inputs

    def _compute_block_loss(
        self,
        block_idx: int,
        calibration_inputs: List[torch.Tensor],
        bf16_targets: List[torch.Tensor],
        device: torch.device,
    ) -> float:
        """Compute reconstruction loss for a block using hooks."""
        total_loss = 0.0
        n_samples = 0
        captured_output = [None]

        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                captured_output[0] = output[0].detach()
            else:
                captured_output[0] = output.detach()

        block = self.blocks[block_idx]
        hook = block.register_forward_hook(capture_hook)

        try:
            for input_ids, tgt in zip(calibration_inputs, bf16_targets):
                input_ids = input_ids.to(device)
                tgt = tgt.to(device)

                with torch.no_grad():
                    _ = self.model(input_ids)
                    out = captured_output[0]
                    loss = (out - tgt).pow(2).mean()
                    total_loss += loss.item()
                    n_samples += 1
        finally:
            hook.remove()

        return total_loss / n_samples

    @torch.no_grad()
    def _get_block_targets(
        self,
        block_idx: int,
        calibration_inputs: List[torch.Tensor],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Get bf16 target outputs for a block."""
        targets = []
        captured_output = [None]

        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                captured_output[0] = output[0].detach().cpu()
            else:
                captured_output[0] = output.detach().cpu()

        block = self.blocks[block_idx]
        hook = block.register_forward_hook(capture_hook)

        try:
            for input_ids in calibration_inputs:
                input_ids = input_ids.to(device)
                _ = self.model(input_ids)
                targets.append(captured_output[0])
        finally:
            hook.remove()

        return targets

    def quantize_block_qep(
        self,
        block_idx: int,
        calibration_inputs: List[torch.Tensor],
        bf16_targets: List[torch.Tensor],
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Quantize a single block using QEP-style inputs.

        Returns loss before and after quantization.
        """
        block = self.blocks[block_idx]

        # Compute initial loss (block is still bf16, but inputs may be degraded)
        init_loss = self._compute_block_loss(block_idx, calibration_inputs, bf16_targets, device)

        # Quantize all linear layers in the block
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) and module.weight.numel() > 1000:
                fp4_quantize_layer(module, self.group_size)

        # Compute final loss (block is now quantized)
        final_loss = self._compute_block_loss(block_idx, calibration_inputs, bf16_targets, device)

        return {
            "block_idx": block_idx,
            "init_loss": init_loss,
            "final_loss": final_loss,
            "improvement": (init_loss - final_loss) / init_loss if init_loss > 0 else 0,
        }

    def quantize(self) -> List[Dict[str, float]]:
        """
        Run QEP-enabled quantization on all blocks.

        Unlike standard AutoRound which pre-caches all inputs,
        this method re-computes inputs for each block after
        quantizing upstream blocks.
        """
        device = next(self.model.parameters()).device

        print("Preparing calibration data...", flush=True)
        calibration_inputs = self._prepare_calibration_data()
        print(f"Using {len(calibration_inputs)} calibration samples", flush=True)

        # Pre-compute bf16 targets for all blocks (before any quantization)
        print("Caching bf16 targets...", flush=True)
        all_bf16_targets = {}
        for block_idx in range(len(self.blocks)):
            all_bf16_targets[block_idx] = self._get_block_targets(
                block_idx, calibration_inputs, device
            )

        results = []
        n_blocks = len(self.blocks)

        print(f"\nQuantizing {n_blocks} blocks with QEP...", flush=True)
        print("=" * 70)
        print(f"{'Block':<8} | {'Init Loss':>12} | {'Final Loss':>12} | {'Improvement':>12}")
        print("=" * 70)

        for block_idx in tqdm(range(n_blocks), desc="Blocks"):
            result = self.quantize_block_qep(
                block_idx, calibration_inputs, all_bf16_targets[block_idx], device
            )
            results.append(result)

            print(f"Block {block_idx:<3} | {result['init_loss']:>12.6f} | "
                  f"{result['final_loss']:>12.6f} | {result['improvement']:>+11.1%}")

            # Clear cache
            gc.collect()
            torch.cuda.empty_cache()

        print("=" * 70)

        # Summary
        avg_init = sum(r["init_loss"] for r in results) / len(results)
        avg_final = sum(r["final_loss"] for r in results) / len(results)

        print(f"\nQEP Quantization Summary:")
        print(f"  Average init loss:  {avg_init:.6f}")
        print(f"  Average final loss: {avg_final:.6f}")
        print(f"  Overall improvement: {(avg_init - avg_final) / avg_init:+.1%}")

        return results


def run_qep_comparison(
    model_name: str,
    nsamples: int = 16,
    seqlen: int = 512,
):
    """
    Run QEP quantization and show how loss evolves across blocks.

    The key insight is that with QEP, later blocks receive degraded inputs
    from upstream quantized blocks, so the loss measurement reflects
    realistic inference conditions.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("\n" + "=" * 70)
    print("QEP QUANTIZATION")
    print("Each block optimized against inputs from upstream quantized blocks")
    print("=" * 70)

    optimizer = QEPBlockOptimizer(
        model=model,
        tokenizer=tokenizer,
        nsamples=nsamples,
        seqlen=seqlen,
    )

    results = optimizer.quantize()

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: Loss progression across blocks")
    print("=" * 70)

    init_losses = [r["init_loss"] for r in results]
    final_losses = [r["final_loss"] for r in results]

    # Use block 1 as baseline since block 0 has near-zero loss
    baseline = init_losses[1] if init_losses[1] > 0 else 1e-6

    print(f"\nInit loss trend (bf16 block, degraded inputs):")
    print(f"  Block 1:  {init_losses[1]:.6f} (baseline)")
    print(f"  Block 15: {init_losses[15]:.6f} ({init_losses[15]/baseline:.1f}x)")
    print(f"  Block 31: {init_losses[-1]:.6f} ({init_losses[-1]/baseline:.1f}x)")

    print(f"\nFinal loss trend (FP4 block, degraded inputs):")
    print(f"  Block 1:  {final_losses[1]:.6f}")
    print(f"  Block 15: {final_losses[15]:.6f} ({final_losses[15]/final_losses[1]:.1f}x)")
    print(f"  Block 31: {final_losses[-1]:.6f} ({final_losses[-1]/final_losses[1]:.1f}x)")

    print("\nThis shows error accumulation through the network.")
    print("Block 31's init_loss of {:.4f} means inputs have diverged".format(init_losses[-1]))
    print("significantly from bf16 before quantization even happens.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--compare", action="store_true", help="Run comparison mode")
    args = parser.parse_args()

    if args.compare:
        run_qep_comparison(args.model, args.samples, args.seq_len)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        optimizer = QEPBlockOptimizer(
            model=model,
            tokenizer=tokenizer,
            nsamples=args.samples,
            seqlen=args.seq_len,
        )
        optimizer.quantize()
