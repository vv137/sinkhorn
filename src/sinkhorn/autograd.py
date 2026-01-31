"""PyTorch autograd integration for Sinkhorn with implicit differentiation."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from sinkhorn.pytorch.implicit_diff import (
    compute_transport_plan,
    conjugate_gradient_solve,
)
from sinkhorn.triton.forward import sinkhorn_forward_triton


class SinkhornFunction(torch.autograd.Function):
    """Autograd function for Sinkhorn with implicit differentiation backward."""

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
        cg_max_iters: int,
        cg_tol: float,
        backend: str,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass: run Sinkhorn iterations.

        Returns:
            f: Row potential (B, N)
            g: Column potential (B, M)
        """
        if backend == "triton":
            f, g, n_iters, converged = sinkhorn_forward_triton(
                C, a, b, epsilon, tau_a, tau_b, mask_a, mask_b, max_iters, threshold
            )
        else:
            # Use PyTorch implementation
            from sinkhorn.pytorch.unbalanced import sinkhorn_unbalanced

            f, g, n_iters, converged = sinkhorn_unbalanced(
                C, a, b, epsilon, tau_a, tau_b, mask_a, mask_b, max_iters, threshold
            )

        # Save for backward
        ctx.save_for_backward(C, a, b, f, g, mask_a, mask_b)
        ctx.epsilon = epsilon
        ctx.tau_a = tau_a
        ctx.tau_b = tau_b
        ctx.cg_max_iters = cg_max_iters
        ctx.cg_tol = cg_tol

        return f, g

    @staticmethod
    def backward(
        ctx: Any,
        grad_f: Tensor,
        grad_g: Tensor,
    ) -> tuple[Tensor | None, ...]:
        """Backward pass via implicit differentiation."""
        C, a, b, f, g, mask_a, mask_b = ctx.saved_tensors
        epsilon = ctx.epsilon
        tau_a = ctx.tau_a
        tau_b = ctx.tau_b
        cg_max_iters = ctx.cg_max_iters
        cg_tol = ctx.cg_tol

        # Compute transport plan
        P = compute_transport_plan(C, f, g, epsilon, mask_a, mask_b)

        # Scaling factors
        rho_a = (
            0.0
            if tau_a is None or tau_a == float("inf")
            else epsilon / (epsilon + tau_a)
        )
        rho_b = (
            0.0
            if tau_b is None or tau_b == float("inf")
            else epsilon / (epsilon + tau_b)
        )

        # Solve adjoint system
        lambda_f, lambda_g, _ = conjugate_gradient_solve(
            grad_f,
            grad_g,
            P,
            epsilon,
            rho_a,
            rho_b,
            mask_a,
            mask_b,
            cg_max_iters,
            cg_tol,
        )

        # Compute gradients
        # ∂L/∂C = -(1/ε) * P * (λ_f[:, None] + λ_g[None, :])
        grad_C = (
            -(1.0 / epsilon) * P * (lambda_f.unsqueeze(-1) + lambda_g.unsqueeze(-2))
        )

        # ∂L/∂a = ε * λ_f (divided by a at application site if needed)
        grad_a = epsilon * lambda_f

        # ∂L/∂b = ε * λ_g
        grad_b = epsilon * lambda_g

        # Return gradients (None for non-tensor args)
        return (
            grad_C,
            grad_a,
            grad_b,
            None,
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
    cg_max_iters: int = 50,
    cg_tol: float = 1e-6,
    backend: str = "triton",
) -> tuple[Tensor, Tensor]:
    """Differentiable Sinkhorn with implicit differentiation backward.

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
        cg_max_iters: Maximum CG iterations for backward
        cg_tol: CG convergence tolerance
        backend: "triton" or "pytorch"

    Returns:
        f: Row potential (B, N)
        g: Column potential (B, M)
    """
    B, N, M = C.shape
    device = C.device
    dtype = C.dtype

    # Create default marginals if needed
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

    # Handle None masks for autograd (convert to dummy tensors)
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
        cg_max_iters,
        cg_tol,
        backend,
    )
