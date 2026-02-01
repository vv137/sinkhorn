"""PyTorch autograd integration for Sinkhorn.

Supports two gradient computation strategies:
1. Full checkpointing (grad_iters=0): Exact gradients, O(NM) memory
2. Truncated backprop (grad_iters>0): Approximate gradients, faster
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from sinkhorn.triton.forward import sinkhorn_forward_triton


class SinkhornFunction(torch.autograd.Function):
    """Autograd function for Sinkhorn with efficient gradient computation."""

    @staticmethod
    def forward(
        ctx: Any,
        C: Tensor,
        a: Tensor,
        b: Tensor,
        epsilon: float,
        tau_a: float | None,
        tau_b: float | None,
        mask_a: Tensor | None,
        mask_b: Tensor | None,
        max_iters: int,
        threshold: float,
        grad_iters: int,
        backend: str,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass: run Sinkhorn iterations without gradient tracking."""
        with torch.no_grad():
            if backend == "triton":
                f, g, n_iters, converged = sinkhorn_forward_triton(
                    C, a, b, epsilon, tau_a, tau_b, mask_a, mask_b, max_iters, threshold
                )
            else:
                from sinkhorn.pytorch.unbalanced import sinkhorn_unbalanced

                f, g, n_iters, converged = sinkhorn_unbalanced(
                    C, a, b, epsilon, tau_a, tau_b, mask_a, mask_b, max_iters, threshold
                )

        ctx.save_for_backward(C, a, b, mask_a, mask_b, f, g)
        ctx.epsilon = epsilon
        ctx.tau_a = tau_a
        ctx.tau_b = tau_b
        ctx.max_iters = max_iters
        ctx.threshold = threshold
        ctx.grad_iters = grad_iters

        return f, g

    @staticmethod
    def backward(
        ctx: Any,
        grad_f: Tensor,
        grad_g: Tensor,
    ) -> tuple[Tensor | None, ...]:
        """Backward pass via recomputation (full or truncated)."""
        C, a, b, mask_a, mask_b, f_conv, g_conv = ctx.saved_tensors
        epsilon = ctx.epsilon
        tau_a = ctx.tau_a
        tau_b = ctx.tau_b
        max_iters = ctx.max_iters
        threshold = ctx.threshold
        grad_iters = ctx.grad_iters

        # Determine number of gradient iterations
        use_truncated = 0 < grad_iters < max_iters

        with torch.enable_grad():
            C_grad = C.detach().requires_grad_(True)
            a_grad = a.detach().requires_grad_(a.requires_grad)
            b_grad = b.detach().requires_grad_(b.requires_grad)

            if use_truncated:
                # Truncated: run few iterations from converged state
                f, g = _sinkhorn_iters_from_init(
                    C_grad,
                    a_grad,
                    b_grad,
                    epsilon,
                    tau_a,
                    tau_b,
                    mask_a,
                    mask_b,
                    f_conv.detach(),
                    g_conv.detach(),
                    grad_iters,
                )
            else:
                # Full recomputation
                from sinkhorn.pytorch.unbalanced import sinkhorn_unbalanced

                f, g, _, _ = sinkhorn_unbalanced(
                    C_grad,
                    a_grad,
                    b_grad,
                    epsilon,
                    tau_a,
                    tau_b,
                    mask_a,
                    mask_b,
                    max_iters,
                    threshold,
                )

            loss = (f * grad_f).sum() + (g * grad_g).sum()
            loss.backward()

        return (
            C_grad.grad,
            a_grad.grad if a.requires_grad else None,
            b_grad.grad if b.requires_grad else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _sinkhorn_iters_from_init(
    C: Tensor,
    a: Tensor,
    b: Tensor,
    epsilon: float,
    tau_a: float | None,
    tau_b: float | None,
    mask_a: Tensor | None,
    mask_b: Tensor | None,
    f_init: Tensor,
    g_init: Tensor,
    n_iters: int,
) -> tuple[Tensor, Tensor]:
    """Run Sinkhorn iterations starting from given potentials."""
    # Need to create new tensors that depend on C for gradient flow
    log_a = torch.log(a.clamp(min=1e-38))
    log_b = torch.log(b.clamp(min=1e-38))

    rho_a = (
        0.0 if tau_a is None or tau_a == float("inf") else epsilon / (epsilon + tau_a)
    )
    rho_b = (
        0.0 if tau_b is None or tau_b == float("inf") else epsilon / (epsilon + tau_b)
    )

    f = f_init
    g = g_init

    for _ in range(n_iters):
        # Row update: f depends on C
        log_sum = torch.logsumexp((g.unsqueeze(-2) - C) / epsilon, dim=-1)
        f_new = rho_a * f + (1 - rho_a) * (-epsilon * log_sum + epsilon * log_a)

        # Column update: g depends on C
        log_sum = torch.logsumexp((f_new.unsqueeze(-1) - C) / epsilon, dim=-2)
        g_new = rho_b * g + (1 - rho_b) * (-epsilon * log_sum + epsilon * log_b)

        f, g = f_new, g_new

    if mask_a is not None:
        f = f.masked_fill(~mask_a, 0.0)
    if mask_b is not None:
        g = g.masked_fill(~mask_b, 0.0)

    return f, g


def sinkhorn_differentiable(
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
    grad_iters: int = 0,
    cg_max_iters: int = 50,
    cg_tol: float = 1e-6,
    backend: str = "triton",
) -> tuple[Tensor, Tensor]:
    """Differentiable Sinkhorn with memory-efficient backward pass.

    Args:
        C: Cost matrix (B, N, M)
        a: Source marginal (B, N). Default: uniform
        b: Target marginal (B, M). Default: uniform
        epsilon: Entropic regularization
        tau_a: KL relaxation for source (None = balanced)
        tau_b: KL relaxation for target (None = balanced)
        mask_a: Row mask (B, N)
        mask_b: Column mask (B, M)
        max_iters: Maximum Sinkhorn iterations
        threshold: Convergence threshold
        grad_iters: Iterations for backward (0 = full recompute, >0 = truncated)
        cg_max_iters: (Unused, for API compatibility)
        cg_tol: (Unused, for API compatibility)
        backend: "triton" or "pytorch"

    Returns:
        f: Row potential (B, N)
        g: Column potential (B, M)
    """
    B, N, M = C.shape
    device = C.device
    dtype = C.dtype

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

    if mask_a is None:
        mask_a = torch.ones(B, N, device=device, dtype=torch.bool)
    if mask_b is None:
        mask_b = torch.ones(B, M, device=device, dtype=torch.bool)

    return SinkhornFunction.apply(
        C,
        a,
        b,
        epsilon,
        tau_a,
        tau_b,
        mask_a,
        mask_b,
        max_iters,
        threshold,
        grad_iters,
        backend,
    )
