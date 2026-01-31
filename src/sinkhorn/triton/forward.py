"""Forward Triton kernels for Sinkhorn iterations.

Implements row/column update kernels with unified balanced/unbalanced support.
"""

from __future__ import annotations

import torch
from torch import Tensor

from sinkhorn.triton.lse_kernel import lse_row, lse_col


def sinkhorn_row_update(
    C: Tensor,
    f: Tensor,
    g: Tensor,
    log_a: Tensor,
    epsilon: float,
    rho: float = 0.0,
    block_size: int = 128,
) -> Tensor:
    """Update row potential: f = ρ·f + (1-ρ)·(softmin + ε·log(a)).

    Args:
        C: Cost matrix (B, N, M)
        f: Current row potential (B, N)
        g: Column potential (B, M)
        log_a: Log of source marginal (B, N)
        epsilon: Regularization parameter
        rho: Scaling factor (0 = balanced)
        block_size: Block size for tiling

    Returns:
        Updated row potential (B, N)
    """
    # lse_row now returns softmin directly: -ε * logsumexp((g - C) / ε)
    softmin = lse_row(C, g, epsilon, block_size)

    # Handle potential inf values from masked entries
    softmin = torch.where(torch.isinf(softmin), torch.zeros_like(softmin), softmin)

    # Full update: f = softmin + ε·log(a)
    f_full = softmin + epsilon * log_a

    # Interpolate for unbalanced
    if rho == 0.0:
        return f_full
    return rho * f + (1 - rho) * f_full


def sinkhorn_col_update(
    C: Tensor,
    f: Tensor,
    g: Tensor,
    log_b: Tensor,
    epsilon: float,
    rho: float = 0.0,
    block_size: int = 128,
) -> Tensor:
    """Update column potential: g = ρ·g + (1-ρ)·(softmin + ε·log(b)).

    Args:
        C: Cost matrix (B, N, M)
        f: Row potential (B, N)
        g: Current column potential (B, M)
        log_b: Log of target marginal (B, M)
        epsilon: Regularization parameter
        rho: Scaling factor (0 = balanced)
        block_size: Block size for tiling

    Returns:
        Updated column potential (B, M)
    """
    softmin = lse_col(C, f, epsilon, block_size)
    softmin = torch.where(torch.isinf(softmin), torch.zeros_like(softmin), softmin)

    g_full = softmin + epsilon * log_b

    if rho == 0.0:
        return g_full
    return rho * g + (1 - rho) * g_full


def sinkhorn_forward_triton(
    C: Tensor,
    a: Tensor | None = None,
    b: Tensor | None = None,
    epsilon: float = 0.1,
    tau_a: float | None = None,
    tau_b: float | None = None,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
    max_iters: int = 100,
    threshold: float = 1e-6,
    block_size: int = 128,
) -> tuple[Tensor, Tensor, int, bool]:
    """Sinkhorn forward pass using numerically stable Triton kernels.

    Args:
        C: Cost matrix (B, N, M)
        a: Source marginal (B, N)
        b: Target marginal (B, M)
        epsilon: Regularization parameter
        tau_a: KL relaxation for source (None = balanced)
        tau_b: KL relaxation for target (None = balanced)
        mask_a: Row mask (B, N)
        mask_b: Column mask (B, M)
        max_iters: Maximum iterations
        threshold: Convergence threshold
        block_size: Block size for Triton kernels

    Returns:
        f: Row potential (B, N)
        g: Column potential (B, M)
        n_iters: Number of iterations
        converged: Whether converged
    """
    B, N, M = C.shape
    device = C.device
    dtype = C.dtype

    # Compute scaling factors
    rho_a = (
        0.0 if tau_a is None or tau_a == float("inf") else epsilon / (epsilon + tau_a)
    )
    rho_b = (
        0.0 if tau_b is None or tau_b == float("inf") else epsilon / (epsilon + tau_b)
    )

    # Default uniform marginals
    if a is None:
        if mask_a is not None:
            a = mask_a.float() / mask_a.float().sum(dim=-1, keepdim=True).clamp(min=1)
        else:
            a = torch.ones(B, N, device=device, dtype=dtype) / N

    if b is None:
        if mask_b is not None:
            b = mask_b.float() / mask_b.float().sum(dim=-1, keepdim=True).clamp(min=1)
        else:
            b = torch.ones(B, M, device=device, dtype=dtype) / M

    log_a = torch.log(a.clamp(min=1e-38))
    log_b = torch.log(b.clamp(min=1e-38))

    # Apply masks to cost matrix
    C_masked = C.clone()
    if mask_a is not None:
        C_masked = C_masked.masked_fill(~mask_a.unsqueeze(-1), float("inf"))
    if mask_b is not None:
        C_masked = C_masked.masked_fill(~mask_b.unsqueeze(-2), float("inf"))

    # Initialize potentials
    f = torch.zeros(B, N, device=device, dtype=dtype)
    g = torch.zeros(B, M, device=device, dtype=dtype)

    converged = False
    n_iters = 0

    for i in range(max_iters):
        f_new = sinkhorn_row_update(C_masked, f, g, log_a, epsilon, rho_a, block_size)
        g_new = sinkhorn_col_update(
            C_masked, f_new, g, log_b, epsilon, rho_b, block_size
        )

        # Apply masks during iteration
        if mask_a is not None:
            f_new = f_new.masked_fill(~mask_a, 0.0)
        if mask_b is not None:
            g_new = g_new.masked_fill(~mask_b, 0.0)

        n_iters = i + 1

        # Check convergence via potential change (every 10 iterations for efficiency)
        if (i + 1) % 10 == 0:
            f_diff = (f_new - f).abs().max()
            g_diff = (g_new - g).abs().max()
            if max(f_diff.item(), g_diff.item()) < threshold:
                f, g = f_new, g_new
                converged = True
                break

        f, g = f_new, g_new

    return f, g, n_iters, converged
