"""Balanced Sinkhorn algorithm in pure PyTorch (log-domain).

This module provides a numerically stable implementation of the balanced
Sinkhorn algorithm using log-domain computations with max-subtraction LSE.
"""

from __future__ import annotations

import torch
from torch import Tensor


def softmin_row(C: Tensor, g: Tensor, epsilon: float) -> Tensor:
    """Compute softmin over columns: -ε * logsumexp(-(C - g) / ε, dim=-1).

    Args:
        C: Cost matrix (B, N, M)
        g: Column potential (B, M)
        epsilon: Regularization parameter

    Returns:
        Softmin values (B, N)
    """
    # -(C - g) / ε = (g - C) / ε
    log_K = (g.unsqueeze(-2) - C) / epsilon  # (B, N, M)
    return -epsilon * torch.logsumexp(log_K, dim=-1)


def softmin_col(C: Tensor, f: Tensor, epsilon: float) -> Tensor:
    """Compute softmin over rows: -ε * logsumexp(-(C - f) / ε, dim=-2).

    Args:
        C: Cost matrix (B, N, M)
        f: Row potential (B, N)
        epsilon: Regularization parameter

    Returns:
        Softmin values (B, M)
    """
    log_K = (f.unsqueeze(-1) - C) / epsilon  # (B, N, M)
    return -epsilon * torch.logsumexp(log_K, dim=-2)


def sinkhorn_balanced(
    C: Tensor,
    a: Tensor | None = None,
    b: Tensor | None = None,
    epsilon: float = 0.1,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
    max_iters: int = 100,
    threshold: float = 1e-6,
) -> tuple[Tensor, Tensor, int, bool]:
    """Balanced Sinkhorn algorithm in log-domain.

    Computes optimal transport dual potentials f, g such that the transport
    plan P_ij = exp((f_i + g_j - C_ij) / ε) satisfies marginal constraints.

    The update equations are:
        f = ε·log(a) + softmin_col(C - g)
        g = ε·log(b) + softmin_row(C - f)

    where softmin_col(X) = -ε·logsumexp(-X/ε, dim=cols)

    Args:
        C: Cost matrix of shape (B, N, M)
        a: Source marginal of shape (B, N). Default: uniform
        b: Target marginal of shape (B, M). Default: uniform
        epsilon: Entropic regularization parameter
        mask_a: Boolean mask for valid source points (B, N)
        mask_b: Boolean mask for valid target points (B, M)
        max_iters: Maximum number of iterations
        threshold: Convergence threshold for marginal error

    Returns:
        f: Row potential (B, N)
        g: Column potential (B, M)
        n_iters: Number of iterations performed
        converged: Whether the algorithm converged

    Example:
        >>> C = torch.randn(8, 64, 64)
        >>> a = torch.softmax(torch.randn(8, 64), dim=-1)
        >>> b = torch.softmax(torch.randn(8, 64), dim=-1)
        >>> f, g, iters, conv = sinkhorn_balanced(C, a, b, epsilon=0.1)
    """
    B, N, M = C.shape
    device = C.device
    dtype = C.dtype

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

    # Clamp for numerical stability and compute log
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
        # Sinkhorn updates in log-domain
        # f = ε·log(a) - ε·logsumexp((g - C) / ε, dim=-1)
        f_new = epsilon * log_a + softmin_row(C_masked, g, epsilon)

        # g = ε·log(b) - ε·logsumexp((f - C) / ε, dim=-2)
        g_new = epsilon * log_b + softmin_col(C_masked, f_new, epsilon)

        # Apply masks
        if mask_a is not None:
            f_new = f_new.masked_fill(~mask_a, 0.0)
        if mask_b is not None:
            g_new = g_new.masked_fill(~mask_b, 0.0)

        # Check convergence via potential change
        f_diff = (f_new - f).abs().max()
        g_diff = (g_new - g).abs().max()

        f, g = f_new, g_new
        n_iters = i + 1

        if max(f_diff.item(), g_diff.item()) < threshold:
            converged = True
            break

    return f, g, n_iters, converged


def compute_transport_plan(
    C: Tensor,
    f: Tensor,
    g: Tensor,
    epsilon: float,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
) -> Tensor:
    """Compute transport plan from dual potentials.

    P_ij = exp((f_i + g_j - C_ij) / ε)

    Args:
        C: Cost matrix (B, N, M)
        f: Row potential (B, N)
        g: Column potential (B, M)
        epsilon: Regularization parameter
        mask_a: Row mask (B, N)
        mask_b: Column mask (B, M)

    Returns:
        Transport plan P (B, N, M)
    """
    log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon

    if mask_a is not None:
        log_P = log_P.masked_fill(~mask_a.unsqueeze(-1), float("-inf"))
    if mask_b is not None:
        log_P = log_P.masked_fill(~mask_b.unsqueeze(-2), float("-inf"))

    return torch.exp(log_P)


def compute_marginal_error(
    C: Tensor,
    f: Tensor,
    g: Tensor,
    a: Tensor,
    b: Tensor,
    epsilon: float,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
) -> Tensor:
    """Compute marginal constraint violation.

    Returns:
        Maximum absolute error in marginals (scalar)
    """
    P = compute_transport_plan(C, f, g, epsilon, mask_a, mask_b)

    row_sum = P.sum(dim=-1)
    col_sum = P.sum(dim=-2)

    if mask_a is not None:
        row_err = (row_sum - a).abs().masked_fill(~mask_a, 0.0).max()
    else:
        row_err = (row_sum - a).abs().max()

    if mask_b is not None:
        col_err = (col_sum - b).abs().masked_fill(~mask_b, 0.0).max()
    else:
        col_err = (col_sum - b).abs().max()

    return torch.max(row_err, col_err)
