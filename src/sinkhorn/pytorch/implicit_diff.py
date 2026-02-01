"""Implicit differentiation for memory-efficient backward pass.

Uses Conjugate Gradient (CG) solver for the adjoint system.
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


def schur_complement_matvec(
    v_f: Tensor,
    v_g: Tensor,
    P: Tensor,
    rho_a: float,
    rho_b: float,
    reg: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Apply the Schur complement matrix for the adjoint system.

    The adjoint system is (I - J_T^T) @ lambda = rhs.

    The Jacobian J_T at the fixed point has structure:
        J_T = [ rho_a * I,          (1-rho_a) * diag(1/r) @ P     ]
              [ (1-rho_b) * diag(1/c) @ P^T,  rho_b * I           ]

    Its transpose:
        J_T^T = [ rho_a * I,          (1-rho_b) * P @ diag(1/c)   ]
                [ (1-rho_a) * P^T @ diag(1/r),  rho_b * I         ]

    So (I - J_T^T) is:
        [ (1-rho_a) * I,           -(1-rho_b) * P @ diag(1/c)   ]
        [ -(1-rho_a) * P^T @ diag(1/r),  (1-rho_b) * I          ]

    We add reg * I for regularization to ensure positive definiteness.

    Args:
        v_f: Input vector f component (B, N)
        v_g: Input vector g component (B, M)
        P: Transport plan (B, N, M)
        rho_a: Unbalanced scaling for source
        rho_b: Unbalanced scaling for target
        reg: Regularization for numerical stability

    Returns:
        Result of (I - J_T^T + reg*I) @ v
    """
    # Row and column sums = marginals
    row_sum = P.sum(dim=-1).clamp(min=1e-30)  # (B, N) = r
    col_sum = P.sum(dim=-2).clamp(min=1e-30)  # (B, M) = c

    # Compute J_T^T @ v
    # (J_T^T @ v)_f = rho_a * v_f + (1-rho_b) * P @ (v_g / c)
    # (J_T^T @ v)_g = rho_b * v_g + (1-rho_a) * P^T @ (v_f / r)
    scaled_v_g = v_g / col_sum
    Pv_g = torch.bmm(P, scaled_v_g.unsqueeze(-1)).squeeze(-1)  # (B, N)

    scaled_v_f = v_f / row_sum
    PTv_f = torch.bmm(P.transpose(-2, -1), scaled_v_f.unsqueeze(-1)).squeeze(
        -1
    )  # (B, M)

    JT_v_f = rho_a * v_f + (1 - rho_b) * Pv_g
    JT_v_g = rho_b * v_g + (1 - rho_a) * PTv_f

    # (I - J_T^T + reg*I) @ v = (1 + reg) * v - J_T^T @ v
    out_f = (1 + reg) * v_f - JT_v_f
    out_g = (1 + reg) * v_g - JT_v_g

    return out_f, out_g


def cg_solve(
    rhs_f: Tensor,
    rhs_g: Tensor,
    P: Tensor,
    rho_a: float = 0.0,
    rho_b: float = 0.0,
    max_iters: int = 50,
    tol: float = 1e-5,
    reg: float = 1e-5,
) -> tuple[Tensor, Tensor, int]:
    """Solve (I - J_T^T + reg*I) @ lambda = rhs using Conjugate Gradient.

    CG converges much faster than Neumann series when the spectral radius
    of J_T is close to 1, which happens in practice for Sinkhorn.

    Args:
        rhs_f: Right-hand side for f (B, N)
        rhs_g: Right-hand side for g (B, M)
        P: Transport plan (B, N, M)
        rho_a: Scaling factor for source (0 for balanced)
        rho_b: Scaling factor for target (0 for balanced)
        max_iters: Maximum CG iterations
        tol: Convergence tolerance on residual
        reg: Regularization for stability (makes system PD)

    Returns:
        lambda_f: Solution for f (B, N)
        lambda_g: Solution for g (B, M)
        n_iters: Number of iterations
    """
    B, N = rhs_f.shape
    M = rhs_g.shape[-1]
    device = rhs_f.device
    dtype = rhs_f.dtype

    def matvec(v_f: Tensor, v_g: Tensor) -> tuple[Tensor, Tensor]:
        return schur_complement_matvec(v_f, v_g, P, rho_a, rho_b, reg)

    def dot(a_f: Tensor, a_g: Tensor, b_f: Tensor, b_g: Tensor) -> Tensor:
        """Batched dot product over (f, g) components."""
        return (a_f * b_f).sum(dim=-1) + (a_g * b_g).sum(dim=-1)

    # Initialize: x = 0, r = b - A @ 0 = b
    x_f = torch.zeros(B, N, device=device, dtype=dtype)
    x_g = torch.zeros(B, M, device=device, dtype=dtype)

    r_f = rhs_f.clone()
    r_g = rhs_g.clone()
    p_f = r_f.clone()
    p_g = r_g.clone()

    r_dot = dot(r_f, r_g, r_f, r_g)  # (B,)
    rhs_norm = r_dot.sqrt().max()

    n_iters = 0
    for i in range(max_iters):
        # A @ p
        Ap_f, Ap_g = matvec(p_f, p_g)

        # alpha = r^T r / p^T A p
        pAp = dot(p_f, p_g, Ap_f, Ap_g)
        alpha = r_dot / pAp.clamp(min=1e-30)

        # x = x + alpha * p
        x_f = x_f + alpha.unsqueeze(-1) * p_f
        x_g = x_g + alpha.unsqueeze(-1) * p_g

        # r = r - alpha * A @ p
        r_f = r_f - alpha.unsqueeze(-1) * Ap_f
        r_g = r_g - alpha.unsqueeze(-1) * Ap_g

        n_iters = i + 1

        # Check convergence
        r_dot_new = dot(r_f, r_g, r_f, r_g)
        residual = r_dot_new.sqrt().max()
        if residual < tol * rhs_norm:
            break

        # beta = r_new^T r_new / r^T r
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
    cg_tol: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute gradients via implicit differentiation.

    Given upstream gradients ∂L/∂f and ∂L/∂g, compute:
        ∂L/∂C, ∂L/∂a, ∂L/∂b

    using the implicit function theorem without storing intermediate iterations.

    The key equation is:
        (I - J_T^T) @ lambda = grad
        ∂L/∂C = -(1/ε) * P * (λ_f[:, None] + λ_g[None, :])

    We use CG to solve the adjoint system, which converges much faster
    than Neumann series when the Jacobian's spectral radius is close to 1.
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

    # Solve adjoint system: (I - J_T^T + reg*I) @ lambda = grad
    # Note: we use grad directly as RHS (the derivation gives this sign)
    lambda_f, lambda_g, n_iters = cg_solve(
        grad_f, grad_g, P, rho_a, rho_b, cg_max_iters, cg_tol
    )

    # Compute gradient w.r.t. C
    # ∂L/∂C_ij = -(1/ε) * P_ij * (λ_f^i + λ_g^j)
    grad_C = -(1.0 / epsilon) * P * (lambda_f.unsqueeze(-1) + lambda_g.unsqueeze(-2))

    # Gradient w.r.t. marginals
    grad_a = epsilon * lambda_f
    grad_b = epsilon * lambda_g

    return grad_C, grad_a, grad_b


# Legacy functions for compatibility
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
    """Jacobian-vector product J_T @ v."""
    row_sum = P.sum(dim=-1).clamp(min=1e-30)
    col_sum = P.sum(dim=-2).clamp(min=1e-30)

    Pv_g = torch.bmm(P, v_g.unsqueeze(-1)).squeeze(-1)
    Pv_f = torch.bmm(P.transpose(-2, -1), v_f.unsqueeze(-1)).squeeze(-1)

    out_f = rho_a * v_f + (1 - rho_a) * Pv_g / row_sum
    out_g = rho_b * v_g + (1 - rho_b) * Pv_f / col_sum

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
    """Transpose-Jacobian-vector product J_T^T @ v."""
    row_sum = P.sum(dim=-1).clamp(min=1e-30)
    col_sum = P.sum(dim=-2).clamp(min=1e-30)

    scaled_v_g = v_g / col_sum
    Pv_g = torch.bmm(P, scaled_v_g.unsqueeze(-1)).squeeze(-1)

    scaled_v_f = v_f / row_sum
    Pv_f = torch.bmm(P.transpose(-2, -1), scaled_v_f.unsqueeze(-1)).squeeze(-1)

    out_f = rho_a * v_f + (1 - rho_b) * Pv_g
    out_g = rho_b * v_g + (1 - rho_a) * Pv_f

    if mask_a is not None:
        out_f = out_f.masked_fill(~mask_a, 0.0)
    if mask_b is not None:
        out_g = out_g.masked_fill(~mask_b, 0.0)

    return out_f, out_g


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
    tol: float = 1e-5,
    regularization: float = 1e-5,
) -> tuple[Tensor, Tensor, int]:
    """Legacy wrapper for cg_solve."""
    return cg_solve(rhs_f, rhs_g, P, rho_a, rho_b, max_iters, tol, regularization)
