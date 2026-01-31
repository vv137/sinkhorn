"""Numerically Stable Streaming Log-Sum-Exp Triton kernel.

Uses the online numerically stable algorithm:
    logsumexp(x) = max(x) + log(sum(exp(x - max(x))))

Key numerical analysis principles applied:
1. Max-subtraction trick: Always subtract max before exp to avoid overflow
2. Welford-style online update: Track (max, sum_exp) pair with stable merging
3. Handle all-negative-infinity case: Check for valid elements before computing
4. Use log1p for precision when possible
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _stable_lse_row_kernel(
    # Pointers
    C_ptr,  # Cost matrix (B, N, M)
    g_ptr,  # Column potential (B, M)
    out_ptr,  # Output: row softmin (B, N)
    # Dimensions
    N,
    M,
    # Parameters
    epsilon,
    # Strides
    stride_cb,  # Batch stride for C
    stride_cn,  # Row stride for C
    stride_cm,  # Column stride for C
    stride_gb,  # Batch stride for g
    stride_gm,  # Element stride for g
    stride_ob,  # Batch stride for out
    stride_on,  # Element stride for out
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
):
    """Compute row-wise softmin: out[n] = -ε * logsumexp_m((g[m] - C[n,m]) / ε).

    Numerically stable online algorithm:
    - For each block, compute local (max, sum_exp)
    - Merge with running (max, sum_exp) using stable formula:
        new_max = max(running_max, block_max)
        new_sum = running_sum * exp(running_max - new_max)
                + block_sum * exp(block_max - new_max)

    Grid: (B, N)
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Initialize: track (max, sum_exp) with special handling for first block
    # Use -inf for max (identity for max operation)
    # Use 0 for sum_exp (identity for sum)
    global_max = tl.full([], -float("inf"), dtype=tl.float32)
    global_sum_exp = tl.full([], 0.0, dtype=tl.float32)

    # Base pointers
    C_row_ptr = C_ptr + pid_b * stride_cb + pid_n * stride_cn
    g_base_ptr = g_ptr + pid_b * stride_gb

    # Process in blocks
    for m_start in range(0, M, BLOCK_SIZE_M):
        m_offs = m_start + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_offs < M

        # Load values (use inf for C out-of-bounds to make x = -inf)
        c_vals = tl.load(
            C_row_ptr + m_offs * stride_cm, mask=m_mask, other=float("inf")
        )
        g_vals = tl.load(
            g_base_ptr + m_offs * stride_gm, mask=m_mask, other=-float("inf")
        )

        # x = (g - C) / epsilon
        # For masked entries: g=-inf, C=inf -> x = (-inf - inf)/eps = -inf (correct!)
        x = (g_vals - c_vals) / epsilon

        # Block statistics (only over valid entries)
        # tl.max/tl.sum over masked values need care
        block_max = tl.max(tl.where(m_mask, x, -float("inf")))

        # Stable exp: x - block_max (0 when x == block_max, -inf when x == -inf)
        # When block_max == -inf, exp(x - block_max) = exp(-inf - (-inf)) would be NaN
        # So we handle this case: if block_max == -inf, skip this block
        is_valid_block = block_max > -float("inf")

        if is_valid_block:
            # Compute sum(exp(x - block_max)) for valid entries
            exp_vals = tl.exp(x - block_max)
            exp_vals = tl.where(m_mask, exp_vals, 0.0)
            block_sum_exp = tl.sum(exp_vals)

            # Merge with global state using numerically stable formula
            # new_max = max(global_max, block_max)
            new_max = tl.maximum(global_max, block_max)

            # Scale factors for the sums
            # global_scale = exp(global_max - new_max), block_scale = exp(block_max - new_max)
            # When global_max == -inf: global_scale = 0 (correct, no contribution)
            # When new_max == global_max: global_scale = 1, block_scale = exp(block_max - global_max)
            global_scale = tl.exp(global_max - new_max)
            block_scale = tl.exp(block_max - new_max)

            # Handle the case where global_max was -inf (first valid block)
            global_scale = tl.where(global_max > -float("inf"), global_scale, 0.0)

            # Update global state
            global_sum_exp = global_sum_exp * global_scale + block_sum_exp * block_scale
            global_max = new_max

    # Final result: softmin = -ε * (max + log(sum_exp))
    # Handle edge case: if no valid entries, global_max = -inf, global_sum_exp = 0
    # In this case, lse = -inf + log(0) = -inf, softmin = -ε * (-inf) = +inf
    # This is actually the correct behavior for softmin of empty set

    # Safe log: avoid log(0)
    safe_sum_exp = tl.maximum(global_sum_exp, 1e-38)
    lse = global_max + tl.log(safe_sum_exp)

    # softmin = -ε * lse
    softmin = -epsilon * lse

    # For completely masked rows (global_max == -inf), return 0 (will be masked later anyway)
    softmin = tl.where(global_max > -float("inf"), softmin, 0.0)

    # Store
    out_offset = pid_b * stride_ob + pid_n * stride_on
    tl.store(out_ptr + out_offset, softmin)


@triton.jit
def _stable_lse_col_kernel(
    # Pointers
    C_ptr,  # Cost matrix (B, N, M)
    f_ptr,  # Row potential (B, N)
    out_ptr,  # Output: col softmin (B, M)
    # Dimensions
    N,
    M,
    # Parameters
    epsilon,
    # Strides
    stride_cb,
    stride_cn,
    stride_cm,
    stride_fb,
    stride_fn,
    stride_ob,
    stride_om,
    # Block sizes
    BLOCK_SIZE_N: tl.constexpr,
):
    """Compute column-wise softmin: out[m] = -ε * logsumexp_n((f[n] - C[n,m]) / ε).

    Grid: (B, M)
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    global_max = tl.full([], -float("inf"), dtype=tl.float32)
    global_sum_exp = tl.full([], 0.0, dtype=tl.float32)

    C_col_ptr = C_ptr + pid_b * stride_cb + pid_m * stride_cm
    f_base_ptr = f_ptr + pid_b * stride_fb

    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offs = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offs < N

        c_vals = tl.load(
            C_col_ptr + n_offs * stride_cn, mask=n_mask, other=float("inf")
        )
        f_vals = tl.load(
            f_base_ptr + n_offs * stride_fn, mask=n_mask, other=-float("inf")
        )

        x = (f_vals - c_vals) / epsilon

        block_max = tl.max(tl.where(n_mask, x, -float("inf")))
        is_valid_block = block_max > -float("inf")

        if is_valid_block:
            exp_vals = tl.exp(x - block_max)
            exp_vals = tl.where(n_mask, exp_vals, 0.0)
            block_sum_exp = tl.sum(exp_vals)

            new_max = tl.maximum(global_max, block_max)
            global_scale = tl.exp(global_max - new_max)
            block_scale = tl.exp(block_max - new_max)
            global_scale = tl.where(global_max > -float("inf"), global_scale, 0.0)

            global_sum_exp = global_sum_exp * global_scale + block_sum_exp * block_scale
            global_max = new_max

    safe_sum_exp = tl.maximum(global_sum_exp, 1e-38)
    lse = global_max + tl.log(safe_sum_exp)
    softmin = -epsilon * lse
    softmin = tl.where(global_max > -float("inf"), softmin, 0.0)

    out_offset = pid_b * stride_ob + pid_m * stride_om
    tl.store(out_ptr + out_offset, softmin)


def lse_row(
    C: Tensor,
    g: Tensor,
    epsilon: float,
    block_size: int = 128,
) -> Tensor:
    """Compute row-wise softmin using numerically stable Triton kernel.

    Args:
        C: Cost matrix (B, N, M)
        g: Column potential (B, M)
        epsilon: Regularization parameter
        block_size: Block size for tiling

    Returns:
        Row softmin values (B, N): -ε * logsumexp_m((g[m] - C[n,m]) / ε)
    """
    B, N, M = C.shape

    out = torch.empty(B, N, device=C.device, dtype=C.dtype)

    grid = (B, N)

    _stable_lse_row_kernel[grid](
        C,
        g,
        out,
        N,
        M,
        epsilon,
        C.stride(0),
        C.stride(1),
        C.stride(2),
        g.stride(0),
        g.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=block_size,
    )

    return out


def lse_col(
    C: Tensor,
    f: Tensor,
    epsilon: float,
    block_size: int = 128,
) -> Tensor:
    """Compute column-wise softmin using numerically stable Triton kernel.

    Args:
        C: Cost matrix (B, N, M)
        f: Row potential (B, N)
        epsilon: Regularization parameter
        block_size: Block size for tiling

    Returns:
        Column softmin values (B, M): -ε * logsumexp_n((f[n] - C[n,m]) / ε)
    """
    B, N, M = C.shape

    out = torch.empty(B, M, device=C.device, dtype=C.dtype)

    grid = (B, M)

    _stable_lse_col_kernel[grid](
        C,
        f,
        out,
        N,
        M,
        epsilon,
        C.stride(0),
        C.stride(1),
        C.stride(2),
        f.stride(0),
        f.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_N=block_size,
    )

    return out
