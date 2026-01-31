"""JVP Triton kernel for implicit differentiation backward pass."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _jvp_row_kernel(
    # Pointers
    P_ptr,  # Transport plan (B, N, M)
    v_g_ptr,  # Input vector (B, M)
    out_ptr,  # Output: P @ v_g / row_sum (B, N)
    # Dimensions
    N,
    M,
    # Strides
    stride_pb,
    stride_pn,
    stride_pm,
    stride_vb,
    stride_vm,
    stride_ob,
    stride_on,
    # Block size
    BLOCK_SIZE_M: tl.constexpr,
):
    """Compute (P @ v_g) / row_sum for each row.

    Grid: (B, N)
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    P_row_ptr = P_ptr + pid_b * stride_pb + pid_n * stride_pn
    v_base_ptr = v_g_ptr + pid_b * stride_vb

    # Accumulate P @ v and row sum
    pv_sum = 0.0
    row_sum = 0.0

    for m_start in range(0, M, BLOCK_SIZE_M):
        m_offs = m_start + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_offs < M

        p_vals = tl.load(P_row_ptr + m_offs * stride_pm, mask=m_mask, other=0.0)
        v_vals = tl.load(v_base_ptr + m_offs * stride_vm, mask=m_mask, other=0.0)

        pv_sum += tl.sum(p_vals * v_vals)
        row_sum += tl.sum(p_vals)

    # Avoid division by zero
    row_sum = tl.maximum(row_sum, 1e-38)
    result = pv_sum / row_sum

    out_offset = pid_b * stride_ob + pid_n * stride_on
    tl.store(out_ptr + out_offset, result)


@triton.jit
def _jvp_col_kernel(
    # Pointers
    P_ptr,  # Transport plan (B, N, M)
    v_f_ptr,  # Input vector (B, N)
    out_ptr,  # Output: P.T @ v_f / col_sum (B, M)
    # Dimensions
    N,
    M,
    # Strides
    stride_pb,
    stride_pn,
    stride_pm,
    stride_vb,
    stride_vn,
    stride_ob,
    stride_om,
    # Block size
    BLOCK_SIZE_N: tl.constexpr,
):
    """Compute (P.T @ v_f) / col_sum for each column.

    Grid: (B, M)
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    P_col_ptr = P_ptr + pid_b * stride_pb + pid_m * stride_pm
    v_base_ptr = v_f_ptr + pid_b * stride_vb

    pv_sum = 0.0
    col_sum = 0.0

    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offs = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offs < N

        p_vals = tl.load(P_col_ptr + n_offs * stride_pn, mask=n_mask, other=0.0)
        v_vals = tl.load(v_base_ptr + n_offs * stride_vn, mask=n_mask, other=0.0)

        pv_sum += tl.sum(p_vals * v_vals)
        col_sum += tl.sum(p_vals)

    col_sum = tl.maximum(col_sum, 1e-38)
    result = pv_sum / col_sum

    out_offset = pid_b * stride_ob + pid_m * stride_om
    tl.store(out_ptr + out_offset, result)


def jvp_triton(
    v_f: Tensor,
    v_g: Tensor,
    P: Tensor,
    rho_a: float = 0.0,
    rho_b: float = 0.0,
    block_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """Compute JVP using Triton kernels.

    J_T @ v = ρ * v + (1 - ρ) * balanced_jvp

    Args:
        v_f: Input vector for f (B, N)
        v_g: Input vector for g (B, M)
        P: Transport plan (B, N, M)
        rho_a: Scaling factor for source
        rho_b: Scaling factor for target
        block_size: Block size for Triton kernels

    Returns:
        JVP result (out_f, out_g)
    """
    B, N, M = P.shape
    device = P.device
    dtype = P.dtype

    out_f = torch.empty(B, N, device=device, dtype=dtype)
    out_g = torch.empty(B, M, device=device, dtype=dtype)

    # Compute balanced JVP parts
    _jvp_row_kernel[(B, N)](
        P,
        v_g,
        out_f,
        N,
        M,
        P.stride(0),
        P.stride(1),
        P.stride(2),
        v_g.stride(0),
        v_g.stride(1),
        out_f.stride(0),
        out_f.stride(1),
        BLOCK_SIZE_M=block_size,
    )

    _jvp_col_kernel[(B, M)](
        P,
        v_f,
        out_g,
        N,
        M,
        P.stride(0),
        P.stride(1),
        P.stride(2),
        v_f.stride(0),
        v_f.stride(1),
        out_g.stride(0),
        out_g.stride(1),
        BLOCK_SIZE_N=block_size,
    )

    # Apply unbalanced scaling
    if rho_a != 0.0:
        out_f = rho_a * v_f + (1 - rho_a) * out_f
    if rho_b != 0.0:
        out_g = rho_b * v_g + (1 - rho_b) * out_g

    return out_f, out_g
