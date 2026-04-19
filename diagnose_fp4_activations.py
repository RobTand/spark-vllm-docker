"""
Diagnose why FP4 activation quantization fails on DeltaNet layers.

Loads the model, runs a forward pass, captures activations at each layer,
quantizes them to FP4 and back, and measures the error. Compares DeltaNet
layers vs self_attn/MLP layers to see if the activation distribution is
fundamentally different.
"""

import torch
import sys

def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    import flashinfer

    MODEL_ID = "Qwen/Qwen3.5-27B"
    print(f"Loading {MODEL_ID}...", file=sys.stderr)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()

    # Hook to capture activations at each linear layer
    activation_stats = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            if not isinstance(x, torch.Tensor) or x.ndim < 2:
                return

            with torch.no_grad():
                x_flat = x.float().reshape(-1)

                # Stats
                stats = {
                    'shape': list(x.shape),
                    'mean': x_flat.mean().item(),
                    'std': x_flat.std().item(),
                    'min': x_flat.min().item(),
                    'max': x_flat.max().item(),
                    'abs_max': x_flat.abs().max().item(),
                    'num_elements': x_flat.numel(),
                }

                # Simulate FP4 quantization and measure error
                # FP4 E2M1 has 16 values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
                # With block scaling (block_size=16):
                # For each block of 16 elements:
                #   scale = max_abs / 6.0 (FP4_MAX)
                #   quantized = round_to_nearest_fp4(x / scale)
                #   dequantized = quantized * scale

                # E2M1 representable values (positive)
                fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                          device=x.device, dtype=torch.float32)
                all_fp4 = torch.cat([-fp4_values.flip(0)[:-1], fp4_values])  # -6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6

                # Reshape to blocks of 16
                numel = x_flat.numel()
                pad_to = ((numel + 15) // 16) * 16
                x_padded = torch.zeros(pad_to, device=x.device, dtype=torch.float32)
                x_padded[:numel] = x_flat
                x_blocks = x_padded.view(-1, 16)

                # Per-block max abs for scaling
                block_max = x_blocks.abs().max(dim=1, keepdim=True).values
                block_max = block_max.clamp(min=1e-10)
                scale = block_max / 6.0

                # Normalize to FP4 range
                x_scaled = x_blocks / scale

                # Round to nearest FP4 value
                x_scaled_flat = x_scaled.reshape(-1, 1)
                distances = (x_scaled_flat - all_fp4.unsqueeze(0)).abs()
                nearest_idx = distances.argmin(dim=1)
                x_quantized = all_fp4[nearest_idx].view_as(x_scaled)

                # Dequantize
                x_dequant = (x_quantized * scale).reshape(-1)[:numel]

                # Error metrics
                error = (x_flat - x_dequant).abs()
                rel_error = error / (x_flat.abs() + 1e-10)

                stats['fp4_mae'] = error.mean().item()
                stats['fp4_max_error'] = error.max().item()
                stats['fp4_rel_mae'] = rel_error.mean().item()
                stats['fp4_snr_db'] = (10 * torch.log10(
                    x_flat.pow(2).mean() / (error.pow(2).mean() + 1e-20)
                )).item()

                # What fraction of values are clipped to ±6 (max FP4)
                stats['fp4_clip_pct'] = ((x_scaled.abs() > 6.0).float().mean() * 100).item()

                # Distribution of quantized values
                unique_vals, counts = x_quantized.reshape(-1).unique(return_counts=True)
                stats['fp4_unique_values'] = len(unique_vals)
                # How concentrated is the distribution?
                probs = counts.float() / counts.sum()
                stats['fp4_entropy'] = -(probs * probs.log2()).sum().item()

            activation_stats[name] = stats
        return hook

    # Register hooks on linear layers inside decoder
    hooks = []
    for name, module in model.named_modules():
        if 'language_model.layers' in name and isinstance(module, torch.nn.Linear):
            # Only hook the first instance of each layer type
            layer_num = name.split('.layers.')[1].split('.')[0]
            if int(layer_num) < 4:  # First 4 layers for speed
                hooks.append(module.register_forward_hook(make_hook(name)))

    # Run a forward pass
    print("Running forward pass...", file=sys.stderr)
    input_text = "The quick brown fox jumps over the lazy dog. " * 10
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print results grouped by layer type
    print("\n" + "=" * 90)
    print("FP4 ACTIVATION QUANTIZATION ERROR ANALYSIS")
    print("=" * 90)

    linear_attn_stats = []
    other_stats = []

    for name, stats in sorted(activation_stats.items()):
        is_deltanet = 'linear_attn' in name
        entry = (name, stats)
        if is_deltanet:
            linear_attn_stats.append(entry)
        else:
            other_stats.append(entry)

    def print_group(title, entries):
        print(f"\n--- {title} ---")
        print(f"{'Layer':<65s} {'AbsMax':>8s} {'Std':>8s} {'FP4 MAE':>10s} {'FP4 Rel%':>9s} {'SNR(dB)':>8s} {'Clip%':>6s} {'Entropy':>8s}")
        for name, s in entries:
            short = name.split('language_model.')[-1]
            print(f"{short:<65s} {s['abs_max']:8.2f} {s['std']:8.4f} {s['fp4_mae']:10.6f} {s['fp4_rel_mae']*100:8.2f}% {s['fp4_snr_db']:8.1f} {s['fp4_clip_pct']:5.1f}% {s['fp4_entropy']:8.2f}")

    print_group("DeltaNet linear_attn layers", linear_attn_stats)
    print_group("Other layers (self_attn, MLP, etc.)", other_stats)

    # Summary comparison
    if linear_attn_stats and other_stats:
        la_snr = sum(s['fp4_snr_db'] for _, s in linear_attn_stats) / len(linear_attn_stats)
        ot_snr = sum(s['fp4_snr_db'] for _, s in other_stats) / len(other_stats)
        la_rel = sum(s['fp4_rel_mae'] for _, s in linear_attn_stats) / len(linear_attn_stats)
        ot_rel = sum(s['fp4_rel_mae'] for _, s in other_stats) / len(other_stats)
        print(f"\n{'='*90}")
        print(f"SUMMARY:")
        print(f"  DeltaNet avg SNR: {la_snr:.1f} dB, avg relative error: {la_rel*100:.2f}%")
        print(f"  Other avg SNR:    {ot_snr:.1f} dB, avg relative error: {ot_rel*100:.2f}%")
        print(f"  SNR difference:   {la_snr - ot_snr:+.1f} dB (negative = DeltaNet is worse)")


if __name__ == "__main__":
    main()
