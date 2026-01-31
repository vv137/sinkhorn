"""Unbalanced Sinkhorn algorithm with unified update rule.

Supports both balanced (τ → ∞) and unbalanced (finite τ) optimal transport
using a unified scaling factor ρ = ε / (ε + τ).
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_rho(epsilon: float, tau: float | None) -> float:
    """Compute scaling factor ρ = ε / (ε + τ).

    Args:
        epsilon: Regularization parameter
        tau: KL relaxation parameter (None or inf means balanced)

    Returns:
        Scaling factor in [0, 1). 0 for balanced, approaching 1 as τ → 0.
    """
    if tau is None or tau == float("inf"):
        return 0.0
    return epsilon / (epsilon + tau)


def softmin_row(C: Tensor, g: Tensor, epsilon: float) -> Tensor:
    """Compute softmin over columns: -ε * logsumexp(-(C - g) / ε, dim=-1)."""
    log_K = (g.unsqueeze(-2) - C) / epsilon
    return -epsilon * torch.logsumexp(log_K, dim=-1)


def softmin_col(C: Tensor, f: Tensor, epsilon: float) -> Tensor:
    """Compute softmin over rows: -ε * logsumexp(-(C - f) / ε, dim=-2)."""
    log_K = (f.unsqueeze(-1) - C) / epsilon
    return -epsilon * torch.logsumexp(log_K, dim=-2)


def sinkhorn_unbalanced(
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
) -> tuple[Tensor, Tensor, int, bool]:
    """Unified Sinkhorn algorithm for balanced and unbalanced OT.

    Uses the unified update rule with scaling factor ρ = ε / (ε + τ):
        f ← ρ·f_old + (1-ρ)·(ε·log(a) + softmin(C - g))
        g ← ρ·g_old + (1-ρ)·(ε·log(b) + softmin(C - f))

    When τ = ∞ (or None), ρ = 0 and this becomes balanced Sinkhorn.

    Args:
        C: Cost matrix of shape (B, N, M)
        a: Source marginal of shape (B, N). Default: uniform
        b: Target marginal of shape (B, M). Default: uniform
        epsilon: Entropic regularization parameter
        tau_a: KL relaxation for source marginal (None = balanced)
        tau_b: KL relaxation for target marginal (None = balanced)
        mask_a: Boolean mask for valid source points (B, N)
        mask_b: Boolean mask for valid target points (B, M)
        max_iters: Maximum number of iterations
        threshold: Convergence threshold for potential change

    Returns:
        f: Row potential (B, N)
        g: Column potential (B, M)
        n_iters: Number of iterations performed
        converged: Whether the algorithm converged

    Example:
        >>> C = torch.randn(8, 64, 64)
        >>> a = torch.softmax(torch.randn(8, 64), dim=-1)
        >>> b = torch.softmax(torch.randn(8, 64), dim=-1)
        >>> # Balanced
        >>> f, g, _, _ = sinkhorn_unbalanced(C, a, b, epsilon=0.1)
        >>> # Unbalanced
        >>> f, g, _, _ = sinkhorn_unbalanced(C, a, b, epsilon=0.1, tau_a=1.0, tau_b=1.0)
    """
    B, N, M = C.shape
    device = C.device
    dtype = C.dtype

    # Compute scaling factors
    rho_a = compute_rho(epsilon, tau_a)
    rho_b = compute_rho(epsilon, tau_b)

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

    # Mask cost matrix
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
        # Full update (balanced case)
        # Note: softmin returns -inf for fully masked rows where all cols are inf
        softmin_f = softmin_row(C_masked, g, epsilon)
        # Replace -inf with 0 to avoid nan when adding to log_a
        softmin_f = torch.where(
            torch.isinf(softmin_f), torch.zeros_like(softmin_f), softmin_f
        )
        f_full = epsilon * log_a + softmin_f

        # Interpolate with previous (unbalanced)
        f_new = rho_a * f + (1 - rho_a) * f_full

        # Same for g
        softmin_g = softmin_col(C_masked, f_new, epsilon)
        softmin_g = torch.where(
            torch.isinf(softmin_g), torch.zeros_like(softmin_g), softmin_g
        )
        g_full = epsilon * log_b + softmin_g
        g_new = rho_b * g + (1 - rho_b) * g_full

        # Apply masks (set masked entries to 0)
        if mask_a is not None:
            f_new = f_new.masked_fill(~mask_a, 0.0)
        if mask_b is not None:
            g_new = g_new.masked_fill(~mask_b, 0.0)

        # Check convergence
        f_diff = (f_new - f).abs().max()
        g_diff = (g_new - g).abs().max()

        f, g = f_new, g_new
        n_iters = i + 1

        if max(f_diff.item(), g_diff.item()) < threshold:
            converged = True
            break

    return f, g, n_iters, converged


def compute_dual_objective(
    C: Tensor,
    f: Tensor,
    g: Tensor,
    a: Tensor,
    b: Tensor,
    epsilon: float,
    tau_a: float | None,
    tau_b: float | None,
    mask_row: Tensor | None = None,
    mask_col: Tensor | None = None,
) -> Tensor:
    """Compute the dual objective for convergence monitoring.

    Returns:
        Dual objective value (B,)
    """
    # <f, a> + <g, b>
    fa = (f * a).sum(dim=-1)
    gb = (g * b).sum(dim=-1)

    # -ε * sum(P)
    log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon
    if mask_row is not None:
        log_P = log_P.masked_fill(~mask_row.unsqueeze(-1), float("-inf"))
    if mask_col is not None:
        log_P = log_P.masked_fill(~mask_col.unsqueeze(-2), float("-inf"))

    P_sum = torch.exp(log_P).sum(dim=(-2, -1))

    return fa + gb - epsilon * P_sum
