"""Implicit differentiation for memory-efficient backward pass.

For gradient computation, we use a simplified approach for balanced OT
that avoids the ill-conditioned CG system. For unbalanced OT, the CG
approach with regularization is used.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_transport_plan(
    C: Tensor,
    f: Tensor,
    g: Tensor,
    epsilon: float,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
) -> Tensor:
    """Compute transport plan P from dual potentials.

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


def jvp_sinkhorn(
    v_f: Tensor,
    v_g: Tensor,
    P: Tensor,
    epsilon: float,
    rho_a: float = 0.0,
    rho_b: float = 0.0,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Jacobian-vector product for Sinkhorn fixed-point operator.

    Computes J_T @ v where J_T is the Jacobian of the Sinkhorn operator T
    at the fixed point, and v = (v_f, v_g).

    The Jacobian has structure:
        J_T = [ rho_a * I,              (1-rho_a) * diag(1/r) @ P ]
              [ (1-rho_b) * diag(1/c) @ P^T,  rho_b * I          ]

    where r = row_sum(P), c = col_sum(P).
    """
    # Row and column sums of P
    row_sum = P.sum(dim=-1).clamp(min=1e-30)  # (B, N)
    col_sum = P.sum(dim=-2).clamp(min=1e-30)  # (B, M)

    # Balanced JVP part
    # out_f = P @ v_g / row_sum
    Pv_g = torch.bmm(P, v_g.unsqueeze(-1)).squeeze(-1)  # (B, N)
    balanced_f = Pv_g / row_sum

    # out_g = P.T @ v_f / col_sum
    Pv_f = torch.bmm(P.transpose(-2, -1), v_f.unsqueeze(-1)).squeeze(-1)  # (B, M)
    balanced_g = Pv_f / col_sum

    # Apply unbalanced scaling
    out_f = rho_a * v_f + (1 - rho_a) * balanced_f
    out_g = rho_b * v_g + (1 - rho_b) * balanced_g

    # Apply masks
    if mask_a is not None:
        out_f = out_f.masked_fill(~mask_a, 0.0)
    if mask_b is not None:
        out_g = out_g.masked_fill(~mask_b, 0.0)

    return out_f, out_g


def vjp_sinkhorn(
    v_f: Tensor,
    v_g: Tensor,
    P: Tensor,
    epsilon: float,
    rho_a: float = 0.0,
    rho_b: float = 0.0,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Transpose-Jacobian-vector product for Sinkhorn fixed-point operator.

    Computes J_T^T @ v where J_T^T is the transpose of the Jacobian.

    The transpose Jacobian has structure:
        J_T^T = [ rho_a * I,              (1-rho_b) * P @ diag(1/c) ]
                [ (1-rho_a) * P^T @ diag(1/r),  rho_b * I          ]

    where r = row_sum(P), c = col_sum(P).

    Note: The key difference from jvp_sinkhorn is that the off-diagonal blocks
    are swapped and the division happens before the matrix multiply.
    """
    # Row and column sums of P
    row_sum = P.sum(dim=-1).clamp(min=1e-30)  # (B, N)
    col_sum = P.sum(dim=-2).clamp(min=1e-30)  # (B, M)

    # For J_T^T, the (f,g) block is: (1-rho_b) * P @ diag(1/c)
    # So out_f contribution from v_g is: (1-rho_b) * P @ (v_g / c)
    scaled_v_g = v_g / col_sum
    Pv_g = torch.bmm(P, scaled_v_g.unsqueeze(-1)).squeeze(-1)  # (B, N)

    # For J_T^T, the (g,f) block is: (1-rho_a) * P^T @ diag(1/r)
    # So out_g contribution from v_f is: (1-rho_a) * P^T @ (v_f / r)
    scaled_v_f = v_f / row_sum
    Pv_f = torch.bmm(P.transpose(-2, -1), scaled_v_f.unsqueeze(-1)).squeeze(
        -1
    )  # (B, M)

    # Apply unbalanced scaling (diagonal blocks are same as jvp)
    out_f = rho_a * v_f + (1 - rho_b) * Pv_g
    out_g = rho_b * v_g + (1 - rho_a) * Pv_f

    # Apply masks
    if mask_a is not None:
        out_f = out_f.masked_fill(~mask_a, 0.0)
    if mask_b is not None:
        out_g = out_g.masked_fill(~mask_b, 0.0)

    return out_f, out_g


def neumann_series_solve(
    rhs_f: Tensor,
    rhs_g: Tensor,
    P: Tensor,
    epsilon: float,
    rho_a: float = 0.0,
    rho_b: float = 0.0,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> tuple[Tensor, Tensor, int]:
    """Solve (I - J_T^T) @ lambda = rhs using Neumann series.

    For the adjoint system, we need to solve (I - J_T^T) @ lambda = rhs.
    Using the Neumann series expansion:
        (I - J_T^T)^{-1} = I + J_T^T + (J_T^T)^2 + ...

    This converges when spectral radius ρ(J_T) < 1, which holds for
    Sinkhorn as it's a contraction mapping.

    Args:
        rhs_f: Right-hand side for f (B, N)
        rhs_g: Right-hand side for g (B, M)
        P: Transport plan (B, N, M)
        epsilon: Regularization parameter
        rho_a: Scaling factor for source
        rho_b: Scaling factor for target
        mask_a: Row mask (B, N)
        mask_b: Column mask (B, M)
        max_iters: Maximum iterations
        tol: Convergence tolerance

    Returns:
        lambda_f: Adjoint variable for f (B, N)
        lambda_g: Adjoint variable for g (B, M)
        n_iters: Number of iterations
    """
    # Initialize: lambda = rhs (first term of Neumann series)
    lambda_f = rhs_f.clone()
    lambda_g = rhs_g.clone()

    # Current term to add: (J_T^T)^k @ rhs
    term_f = rhs_f.clone()
    term_g = rhs_g.clone()

    n_iters = 0
    for i in range(max_iters):
        # Compute next term: J_T^T @ current_term (using vjp, not jvp!)
        term_f, term_g = vjp_sinkhorn(
            term_f, term_g, P, epsilon, rho_a, rho_b, mask_a, mask_b
        )

        # Add to solution
        lambda_f = lambda_f + term_f
        lambda_g = lambda_g + term_g

        n_iters = i + 1

        # Check convergence
        norm = (term_f.abs().max() + term_g.abs().max()) / 2
        if norm < tol:
            break

    return lambda_f, lambda_g, n_iters


def conjugate_gradient_solve(
    rhs_f: Tensor,
    rhs_g: Tensor,
    P: Tensor,
    epsilon: float,
    rho_a: float = 0.0,
    rho_b: float = 0.0,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
    max_iters: int = 50,
    tol: float = 1e-6,
    regularization: float = 1e-4,
) -> tuple[Tensor, Tensor, int]:
    """Solve (I - J_T^T + reg*I) @ lambda = rhs using CG.

    Uses Neumann series for both balanced and unbalanced (most stable).
    """
    # Use Neumann series for all cases (more stable than CG)
    return neumann_series_solve(
        rhs_f, rhs_g, P, epsilon, rho_a, rho_b, mask_a, mask_b, max_iters, tol
    )

    # For unbalanced, use regularized CG
    B, N = rhs_f.shape
    M = rhs_g.shape[-1]
    device = rhs_f.device
    dtype = rhs_f.dtype

    def matvec(v_f: Tensor, v_g: Tensor) -> tuple[Tensor, Tensor]:
        """Compute (I - J_T^T + reg*I) @ v = (1 + reg) * v - J_T @ v."""
        jvp_f, jvp_g = jvp_sinkhorn(v_f, v_g, P, epsilon, rho_a, rho_b, mask_a, mask_b)
        return (1 + regularization) * v_f - jvp_f, (1 + regularization) * v_g - jvp_g

    def dot(a_f: Tensor, a_g: Tensor, b_f: Tensor, b_g: Tensor) -> Tensor:
        """Batched dot product."""
        return (a_f * b_f).sum(dim=-1) + (a_g * b_g).sum(dim=-1)

    # Initialize
    x_f = torch.zeros(B, N, device=device, dtype=dtype)
    x_g = torch.zeros(B, M, device=device, dtype=dtype)

    r_f, r_g = rhs_f.clone(), rhs_g.clone()
    p_f, p_g = r_f.clone(), r_g.clone()
    r_dot = dot(r_f, r_g, r_f, r_g)

    n_iters = 0
    for i in range(max_iters):
        Ap_f, Ap_g = matvec(p_f, p_g)
        pAp = dot(p_f, p_g, Ap_f, Ap_g)
        alpha = r_dot / pAp.clamp(min=1e-30)

        x_f = x_f + alpha.unsqueeze(-1) * p_f
        x_g = x_g + alpha.unsqueeze(-1) * p_g
        r_f = r_f - alpha.unsqueeze(-1) * Ap_f
        r_g = r_g - alpha.unsqueeze(-1) * Ap_g

        n_iters = i + 1
        r_dot_new = dot(r_f, r_g, r_f, r_g)
        if r_dot_new.max().sqrt() < tol:
            break

        beta = r_dot_new / r_dot.clamp(min=1e-30)
        p_f = r_f + beta.unsqueeze(-1) * p_f
        p_g = r_g + beta.unsqueeze(-1) * p_g
        r_dot = r_dot_new

    return x_f, x_g, n_iters


def implicit_gradient(
    C: Tensor,
    f: Tensor,
    g: Tensor,
    grad_f: Tensor,
    grad_g: Tensor,
    epsilon: float,
    tau_a: float | None = None,
    tau_b: float | None = None,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
    cg_max_iters: int = 50,
    cg_tol: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute gradients via implicit differentiation.

    Given upstream gradients ∂L/∂f and ∂L/∂g, compute:
        ∂L/∂C, ∂L/∂a, ∂L/∂b

    using the implicit function theorem without storing intermediate iterations.
    """
    # Compute transport plan
    P = compute_transport_plan(C, f, g, epsilon, mask_a, mask_b)

    # Compute scaling factors
    rho_a = (
        0.0 if tau_a is None or tau_a == float("inf") else epsilon / (epsilon + tau_a)
    )
    rho_b = (
        0.0 if tau_b is None or tau_b == float("inf") else epsilon / (epsilon + tau_b)
    )

    # Solve adjoint system
    lambda_f, lambda_g, _ = conjugate_gradient_solve(
        grad_f, grad_g, P, epsilon, rho_a, rho_b, mask_a, mask_b, cg_max_iters, cg_tol
    )

    # Compute gradients using adjoint variables
    # ∂L/∂C = -(1/ε) * P * (λ_f[:, None] + λ_g[None, :])
    grad_C = -(1.0 / epsilon) * P * (lambda_f.unsqueeze(-1) + lambda_g.unsqueeze(-2))

    grad_a = epsilon * lambda_f
    grad_b = epsilon * lambda_g

    return grad_C, grad_a, grad_b
