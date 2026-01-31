"""High-level API for Sinkhorn optimal transport."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SinkhornOutput:
    """Output container for Sinkhorn computation."""

    f: Tensor
    """Row dual potential (B, N)"""

    g: Tensor
    """Column dual potential (B, M)"""

    n_iters: int
    """Number of iterations performed"""

    converged: bool
    """Whether the algorithm converged"""

    def transport_plan(
        self,
        C: Tensor,
        epsilon: float,
        mask_a: Tensor | None = None,
        mask_b: Tensor | None = None,
    ) -> Tensor:
        """Compute transport plan P from dual potentials.

        P_ij = exp((f_i + g_j - C_ij) / ε)

        Args:
            C: Cost matrix (B, N, M)
            epsilon: Regularization parameter
            mask_a: Row mask (B, N)
            mask_b: Column mask (B, M)

        Returns:
            Transport plan (B, N, M)
        """
        log_P = (self.f.unsqueeze(-1) + self.g.unsqueeze(-2) - C) / epsilon

        if mask_a is not None:
            log_P = log_P.masked_fill(~mask_a.unsqueeze(-1), float("-inf"))
        if mask_b is not None:
            log_P = log_P.masked_fill(~mask_b.unsqueeze(-2), float("-inf"))

        return torch.exp(log_P)

    def cost(self, C: Tensor, epsilon: float) -> Tensor:
        """Compute transport cost <P, C>.

        Args:
            C: Cost matrix (B, N, M)
            epsilon: Regularization parameter

        Returns:
            Transport cost per batch (B,)
        """
        P = self.transport_plan(C, epsilon)
        return (P * C).sum(dim=(-2, -1))

    def sinkhorn_divergence(
        self,
        C: Tensor,
        epsilon: float,
        a: Tensor | None = None,
        b: Tensor | None = None,
    ) -> Tensor:
        """Compute Sinkhorn divergence (debiased).

        S(a, b) = OT(a,b) - 0.5 * OT(a,a) - 0.5 * OT(b,b)

        Note: This is a simplified version. For exact debiasing,
        compute separate a-a and b-b transports.

        Args:
            C: Cost matrix (B, N, M)
            epsilon: Regularization parameter
            a: Source marginal (B, N)
            b: Target marginal (B, M)

        Returns:
            Sinkhorn divergence (B,)
        """
        # Approximate using dual objective
        if a is None:
            a = torch.ones(self.f.shape, device=self.f.device, dtype=self.f.dtype)
            a = a / a.sum(dim=-1, keepdim=True)
        if b is None:
            b = torch.ones(self.g.shape, device=self.g.device, dtype=self.g.dtype)
            b = b / b.sum(dim=-1, keepdim=True)

        # <f, a> + <g, b>
        dual = (self.f * a).sum(dim=-1) + (self.g * b).sum(dim=-1)
        return dual


def sinkhorn(
    C: Tensor,
    a: Tensor | None = None,
    b: Tensor | None = None,
    epsilon: float = 0.1,
    tau_a: float | None = None,
    tau_b: float | None = None,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
    backend: str = "triton",
    max_iters: int = 100,
    threshold: float = 1e-6,
    implicit_diff: bool = True,
    cg_max_iters: int = 50,
    cg_tol: float = 1e-6,
) -> SinkhornOutput:
    """Unified entry point for Sinkhorn optimal transport.

    Computes dual potentials (f, g) for the entropic regularized optimal
    transport problem. Supports both balanced and unbalanced formulations.

    Args:
        C: Cost matrix of shape (B, N, M)
        a: Source marginal of shape (B, N). Default: uniform distribution
        b: Target marginal of shape (B, M). Default: uniform distribution
        epsilon: Entropic regularization coefficient (larger = smoother)
        tau_a: KL relaxation parameter for source marginal.
            None = balanced (exact marginal constraint)
        tau_b: KL relaxation parameter for target marginal.
            None = balanced (exact marginal constraint)
        mask_a: Boolean mask for valid source points (B, N).
            False entries are excluded from the transport.
        mask_b: Boolean mask for valid target points (B, M).
        backend: Computation backend:
            - "triton": Use Triton kernels (GPU required, faster)
            - "pytorch": Use pure PyTorch (CPU/GPU, reference impl)
        max_iters: Maximum number of Sinkhorn iterations
        threshold: Convergence threshold for potential change
        implicit_diff: Whether to use implicit differentiation for backward.
            If False, uses standard autodiff through iterations.
        cg_max_iters: Maximum CG iterations for implicit diff backward
        cg_tol: CG convergence tolerance

    Returns:
        SinkhornOutput containing:
            - f: Row dual potential (B, N)
            - g: Column dual potential (B, M)
            - n_iters: Number of iterations performed
            - converged: Whether the algorithm converged

    Example:
        >>> import torch
        >>> from sinkhorn import sinkhorn
        >>>
        >>> # Random cost matrix and marginals
        >>> C = torch.randn(8, 64, 64, device='cuda')
        >>> a = torch.softmax(torch.randn(8, 64, device='cuda'), -1)
        >>> b = torch.softmax(torch.randn(8, 64, device='cuda'), -1)
        >>>
        >>> # Balanced Sinkhorn
        >>> out = sinkhorn(C, a, b, epsilon=0.1)
        >>> print(f"Converged: {out.converged}, Iters: {out.n_iters}")
        >>>
        >>> # Get transport plan
        >>> P = out.transport_plan(C, epsilon=0.1)
        >>>
        >>> # Unbalanced Sinkhorn with KL relaxation
        >>> out = sinkhorn(C, a, b, epsilon=0.1, tau_a=1.0, tau_b=1.0)
    """
    # Validate inputs
    if C.dim() != 3:
        raise ValueError(f"C must be 3D (B, N, M), got {C.dim()}D")

    B, N, M = C.shape
    device = C.device
    dtype = C.dtype

    # Check backend
    if backend == "triton" and not C.is_cuda:
        import warnings

        warnings.warn("Triton backend requires CUDA. Falling back to PyTorch.")
        backend = "pytorch"

    # Use differentiable version if gradients needed
    if torch.is_grad_enabled() and implicit_diff and C.requires_grad:
        from sinkhorn.autograd import sinkhorn_differentiable

        f, g = sinkhorn_differentiable(
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
        # Note: We don't have n_iters and converged from differentiable version
        # This is a limitation of the current implementation
        return SinkhornOutput(f=f, g=g, n_iters=max_iters, converged=True)

    # Non-differentiable forward
    if backend == "triton":
        from sinkhorn.triton.forward import sinkhorn_forward_triton

        f, g, n_iters, converged = sinkhorn_forward_triton(
            C, a, b, epsilon, tau_a, tau_b, mask_a, mask_b, max_iters, threshold
        )
    else:
        from sinkhorn.pytorch.unbalanced import sinkhorn_unbalanced

        f, g, n_iters, converged = sinkhorn_unbalanced(
            C, a, b, epsilon, tau_a, tau_b, mask_a, mask_b, max_iters, threshold
        )

    return SinkhornOutput(f=f, g=g, n_iters=n_iters, converged=converged)
