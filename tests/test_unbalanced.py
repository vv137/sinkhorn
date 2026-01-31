"""Tests for unbalanced Sinkhorn algorithm."""

import pytest
import torch
import math

from sinkhorn.pytorch.unbalanced import sinkhorn_unbalanced, compute_rho


class TestComputeRho:
    """Test scaling factor computation."""

    def test_balanced_rho(self):
        """Test ρ = 0 for balanced (τ = ∞)."""
        assert compute_rho(0.1, None) == 0.0
        assert compute_rho(0.1, float("inf")) == 0.0

    def test_unbalanced_rho(self):
        """Test ρ = ε / (ε + τ) for finite τ."""
        epsilon = 0.1
        tau = 1.0
        expected = epsilon / (epsilon + tau)
        assert math.isclose(compute_rho(epsilon, tau), expected)

    def test_extreme_tau(self):
        """Test extreme τ values."""
        epsilon = 0.1

        # Very large τ → ρ ≈ 0
        assert compute_rho(epsilon, 1000.0) < 0.001

        # Very small τ → ρ ≈ 1
        assert compute_rho(epsilon, 0.001) > 0.99


class TestUnbalancedSinkhorn:
    """Test unbalanced Sinkhorn algorithm."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_balanced_mode(self, device):
        """Test that τ=None gives balanced behavior."""
        B, N, M = 2, 16, 16
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        f, g, _, converged = sinkhorn_unbalanced(
            C, a, b, epsilon=0.1, tau_a=None, tau_b=None, max_iters=200, threshold=1e-6
        )

        assert converged

        # Check marginal constraints are satisfied (balanced)
        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / 0.1
        P = torch.exp(log_P)

        row_err = (P.sum(dim=-1) - a).abs().max()
        col_err = (P.sum(dim=-2) - b).abs().max()

        assert row_err < 1e-3, f"Row marginal error: {row_err}"
        assert col_err < 1e-3, f"Column marginal error: {col_err}"

    def test_unbalanced_relaxation(self, device):
        """Test that unbalanced mode relaxes marginal constraints."""
        B, N, M = 2, 16, 16

        # Create mismatched marginals (different total mass)
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.ones(B, N, device=device) / N
        b = torch.ones(B, M, device=device) / M * 2  # Double mass
        b = b / b.sum(dim=-1, keepdim=True)  # Normalize

        f, g, _, converged = sinkhorn_unbalanced(
            C, a, b, epsilon=0.1, tau_a=1.0, tau_b=1.0, max_iters=200
        )

        assert converged
        assert not torch.isnan(f).any()
        assert not torch.isnan(g).any()

    def test_symmetry(self, device):
        """Test asymmetric τ values."""
        B, N, M = 2, 16, 16
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        # τ_a strict, τ_b relaxed
        f1, g1, _, _ = sinkhorn_unbalanced(C, a, b, epsilon=0.1, tau_a=10.0, tau_b=0.5)

        # τ_a relaxed, τ_b strict
        f2, g2, _, _ = sinkhorn_unbalanced(C, a, b, epsilon=0.1, tau_a=0.5, tau_b=10.0)

        # Results should be different
        assert (f1 - f2).abs().max() > 0.01 or (g1 - g2).abs().max() > 0.01

    def test_convergence_different_tau(self, device):
        """Test convergence with various τ values."""
        B, N, M = 2, 16, 16
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        tau_values = [0.1, 0.5, 1.0, 5.0, 10.0]

        for tau in tau_values:
            f, g, n_iters, converged = sinkhorn_unbalanced(
                C, a, b, epsilon=0.1, tau_a=tau, tau_b=tau, max_iters=200
            )

            assert not torch.isnan(f).any(), f"NaN with τ={tau}"
            assert not torch.isnan(g).any(), f"NaN with τ={tau}"

    def test_transport_nonnegative(self, device):
        """Test that transport plan is non-negative."""
        B, N, M = 2, 16, 16
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        f, g, _, _ = sinkhorn_unbalanced(C, a, b, epsilon=0.1, tau_a=1.0, tau_b=1.0)

        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / 0.1
        P = torch.exp(log_P)

        assert (P >= 0).all(), "Transport plan has negative entries"


class TestMasking:
    """Test masking functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_basic_masking(self, device):
        """Test basic masking functionality."""
        B, N, M = 2, 16, 16
        C = torch.randn(B, N, M, device=device).abs()

        # Mask first half of source
        mask_a = torch.zeros(B, N, device=device, dtype=torch.bool)
        mask_a[:, N // 2 :] = True

        # Mask last half of target
        mask_b = torch.zeros(B, M, device=device, dtype=torch.bool)
        mask_b[:, : M // 2] = True

        # Create marginals only over valid entries
        a = torch.zeros(B, N, device=device)
        a[:, N // 2 :] = 1.0 / (N // 2)

        b = torch.zeros(B, M, device=device)
        b[:, : M // 2] = 1.0 / (M // 2)

        f, g, _, converged = sinkhorn_unbalanced(
            C,
            a,
            b,
            epsilon=0.1,
            tau_a=None,
            tau_b=None,
            mask_a=mask_a,
            mask_b=mask_b,
            max_iters=200,
        )

        # Masked entries should be zero
        assert (f[:, : N // 2].abs() < 1e-6).all(), "Masked f should be near zero"
        assert (g[:, M // 2 :].abs() < 1e-6).all(), "Masked g should be near zero"

    def test_variable_length_batches(self, device):
        """Test variable length sequences in batch."""
        B, N, M = 3, 16, 16
        C = torch.randn(B, N, M, device=device).abs()

        # Different valid lengths per batch
        mask_a = torch.zeros(B, N, device=device, dtype=torch.bool)
        mask_a[0, :8] = True  # Batch 0: 8 valid
        mask_a[1, :12] = True  # Batch 1: 12 valid
        mask_a[2, :16] = True  # Batch 2: 16 valid

        mask_b = torch.zeros(B, M, device=device, dtype=torch.bool)
        mask_b[0, :8] = True
        mask_b[1, :12] = True
        mask_b[2, :16] = True

        # Uniform marginals over valid entries
        a = mask_a.float() / mask_a.float().sum(dim=-1, keepdim=True).clamp(min=1)
        b = mask_b.float() / mask_b.float().sum(dim=-1, keepdim=True).clamp(min=1)

        f, g, _, _ = sinkhorn_unbalanced(
            C, a, b, epsilon=0.1, mask_a=mask_a, mask_b=mask_b, max_iters=100
        )

        # Check that valid entries don't have NaN
        for i in range(B):
            valid_f = f[i][mask_a[i]]
            valid_g = g[i][mask_b[i]]
            assert not torch.isnan(valid_f).any(), (
                f"Batch {i} f has NaN in valid entries"
            )
            assert not torch.isnan(valid_g).any(), (
                f"Batch {i} g has NaN in valid entries"
            )

    def test_mask_excludes_transport(self, device):
        """Test that masked entries have near-zero transport."""
        B, N, M = 2, 16, 16
        C = torch.randn(B, N, M, device=device).abs()

        mask_a = torch.ones(B, N, device=device, dtype=torch.bool)
        mask_a[:, 0] = False  # First row invalid

        mask_b = torch.ones(B, M, device=device, dtype=torch.bool)
        mask_b[:, 0] = False  # First column invalid

        a = torch.ones(B, N, device=device)
        a[:, 0] = 0
        a = a / a.sum(dim=-1, keepdim=True)

        b = torch.ones(B, M, device=device)
        b[:, 0] = 0
        b = b / b.sum(dim=-1, keepdim=True)

        f, g, _, _ = sinkhorn_unbalanced(
            C, a, b, epsilon=0.1, mask_a=mask_a, mask_b=mask_b, max_iters=100
        )

        # Compute transport plan with masking applied
        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / 0.1

        # Apply masks to log_P
        log_P = log_P.masked_fill(~mask_a.unsqueeze(-1), float("-inf"))
        log_P = log_P.masked_fill(~mask_b.unsqueeze(-2), float("-inf"))

        P = torch.exp(log_P)

        # First row and column should have zero transport
        assert P[:, 0, :].abs().max() < 1e-10, "Masked row should have no transport"
        assert P[:, :, 0].abs().max() < 1e-10, "Masked col should have no transport"
