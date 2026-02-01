"""PyTorch autograd integration for Sinkhorn with efficient gradient computation.

Uses checkpointed recomputation for memory-efficient backward pass.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from sinkhorn.triton.forward import sinkhorn_forward_triton


class SinkhornFunction(torch.autograd.Function):
    """Autograd function for Sinkhorn with checkpointed recomputation.

    Forward pass runs without gradient tracking for memory efficiency.
    Backward pass recomputes the forward pass with gradients enabled.
    This gives O(NM) memory instead of O(K*NM) for K iterations, while
    still providing exact gradients.
    """

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
        """Forward pass: run Sinkhorn iterations without gradient tracking.

        Returns:
            f: Row potential (B, N)
            g: Column potential (B, M)
        """
        # Run forward without gradients for memory efficiency
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

        # Save inputs for backward (not intermediate states)
        ctx.save_for_backward(C, a, b, mask_a, mask_b)
        ctx.epsilon = epsilon
        ctx.tau_a = tau_a
        ctx.tau_b = tau_b
        ctx.max_iters = max_iters
        ctx.threshold = threshold

        return f, g

    @staticmethod
    def backward(
        ctx: Any,
        grad_f: Tensor,
        grad_g: Tensor,
    ) -> tuple[Tensor | None, ...]:
        """Backward pass via checkpointed recomputation.

        Re-runs the forward pass with gradients enabled, which is more
        memory-efficient than storing all intermediate states but still
        gives exact gradients.
        """
        C, a, b, mask_a, mask_b = ctx.saved_tensors
        epsilon = ctx.epsilon
        tau_a = ctx.tau_a
        tau_b = ctx.tau_b
        max_iters = ctx.max_iters
        threshold = ctx.threshold

        # Re-run forward with gradients enabled
        with torch.enable_grad():
            C_grad = C.detach().requires_grad_(True)
            a_grad = a.detach().requires_grad_(a.requires_grad)
            b_grad = b.detach().requires_grad_(b.requires_grad)

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

            # Compute weighted loss using upstream gradients
            loss = (f * grad_f).sum() + (g * grad_g).sum()
            loss.backward()

        grad_C = C_grad.grad
        grad_a = a_grad.grad if a.requires_grad else None
        grad_b = b_grad.grad if b.requires_grad else None

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
    """Differentiable Sinkhorn with memory-efficient backward pass.

    Uses checkpointed recomputation: the forward pass runs without storing
    intermediate states, and the backward pass recomputes them on-the-fly.
    This gives O(NM) memory instead of O(K*NM) for K iterations.

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
        cg_max_iters: (Unused, kept for API compatibility)
        cg_tol: (Unused, kept for API compatibility)
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
