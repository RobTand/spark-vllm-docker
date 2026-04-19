"""
Triton FP4 GEMV kernel for M=1 decode.
Placed in /workspace/ inside the container by the mod run.sh.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_gemv_kernel(
    x_ptr, w_ptr, w_sf_ptr, alpha_ptr, out_ptr,
    stride_w: tl.constexpr, stride_sf: tl.constexpr,
    N, K,
    SF_SIZE: tl.constexpr, BLOCK_K_BYTES: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= N:
        return

    acc = tl.zeros([], dtype=tl.float32)
    w_row_ptr = w_ptr + col * stride_w
    sf_row_ptr = w_sf_ptr + col * stride_sf

    for byte_start in range(0, K // 2, BLOCK_K_BYTES):
        byte_offs = byte_start + tl.arange(0, BLOCK_K_BYTES)
        byte_mask = byte_offs < (K // 2)

        w_bytes = tl.load(w_row_ptr + byte_offs, mask=byte_mask, other=0).to(tl.int32)
        lo = w_bytes & 0xF
        hi = (w_bytes >> 4) & 0xF

        # E2M1 dequant
        lo_exp = (lo >> 1) & 0x3
        lo_man = lo & 1
        lo_sign = (lo >> 3) & 1
        lo_normal = tl.exp2((lo_exp - 1).to(tl.float32)) * (1.0 + 0.5 * lo_man.to(tl.float32))
        lo_sub = 0.5 * lo_man.to(tl.float32)
        lo_abs = tl.where(lo_exp > 0, lo_normal, lo_sub)
        lo_f = tl.where(lo_sign > 0, -lo_abs, lo_abs)

        hi_exp = (hi >> 1) & 0x3
        hi_man = hi & 1
        hi_sign = (hi >> 3) & 1
        hi_normal = tl.exp2((hi_exp - 1).to(tl.float32)) * (1.0 + 0.5 * hi_man.to(tl.float32))
        hi_sub = 0.5 * hi_man.to(tl.float32)
        hi_abs = tl.where(hi_exp > 0, hi_normal, hi_sub)
        hi_f = tl.where(hi_sign > 0, -hi_abs, hi_abs)

        # Load input BF16
        k_lo = byte_start * 2 + tl.arange(0, BLOCK_K_BYTES) * 2
        k_hi = k_lo + 1
        x_lo = tl.load(x_ptr + k_lo, mask=byte_mask, other=0.0).to(tl.float32)
        x_hi = tl.load(x_ptr + k_hi, mask=byte_mask, other=0.0).to(tl.float32)

        # FP8 E4M3 block scale decode
        sf_offs = (byte_start * 2 + tl.arange(0, BLOCK_K_BYTES) * 2) // SF_SIZE
        sf_raw = tl.load(sf_row_ptr + sf_offs, mask=byte_mask, other=0).to(tl.int32)
        sf_exp = (sf_raw >> 3) & 0xF
        sf_man = sf_raw & 0x7
        sf_vals = tl.exp2((sf_exp - 7).to(tl.float32)) * (1.0 + sf_man.to(tl.float32) / 8.0)
        sf_vals = tl.where(sf_exp > 0, sf_vals,
                           sf_man.to(tl.float32) / 8.0 * tl.exp2(tl.full(sf_exp.shape, -6.0, dtype=tl.float32)))

        acc += tl.sum(lo_f * x_lo * sf_vals + hi_f * x_hi * sf_vals)

    alpha = tl.load(alpha_ptr)
    acc *= alpha
    tl.store(out_ptr + col, acc.to(tl.bfloat16))


def fp4_gemv(x, w, w_sf, alpha, out=None):
    """FP4 GEMV: y = x @ dequant(W)^T for M=1.
    x: (1, K) BF16, w: (N, K//2) uint8, w_sf: (N, K//16) uint8, alpha: scalar f32
    """
    K = x.shape[1]
    N = w.shape[0]
    if out is None:
        out = torch.empty(N, dtype=torch.bfloat16, device=x.device)

    BLOCK_K_BYTES = min(K // 2, 512)
    _fp4_gemv_kernel[(N,)](
        x.view(-1), w, w_sf, alpha, out,
        stride_w=K // 2, stride_sf=K // 16,
        N=N, K=K, SF_SIZE=16, BLOCK_K_BYTES=BLOCK_K_BYTES,
        num_warps=4,
    )
    return out.unsqueeze(0)
