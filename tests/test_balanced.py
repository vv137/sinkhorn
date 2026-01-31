"""Tests for balanced Sinkhorn algorithm."""

import pytest
import torch

from sinkhorn.pytorch.balanced import sinkhorn_balanced


class TestBalancedSinkhorn:
    """Test balanced Sinkhorn algorithm."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def simple_problem(self, device):
        """Simple OT problem with known solution."""
        B, N, M = 2, 8, 8
        # Identity-like cost matrix
        C = torch.ones(B, N, M, device=device) * 10
        for i in range(min(N, M)):
            C[:, i, i] = 0.0

        # Uniform marginals
        a = torch.ones(B, N, device=device) / N
        b = torch.ones(B, M, device=device) / M

        return C, a, b

    def test_convergence(self, simple_problem):
        """Test that algorithm converges."""
        C, a, b = simple_problem

        f, g, n_iters, converged = sinkhorn_balanced(
            C, a, b, epsilon=0.1, max_iters=100, threshold=1e-6
        )

        assert converged, f"Failed to converge in {n_iters} iterations"
        assert n_iters < 100, f"Took too many iterations: {n_iters}"

    def test_marginal_constraints(self, simple_problem):
        """Test that marginal constraints are satisfied."""
        C, a, b = simple_problem
        epsilon = 0.1

        f, g, _, _ = sinkhorn_balanced(
            C, a, b, epsilon=epsilon, max_iters=200, threshold=1e-8
        )

        # Compute transport plan
        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon
        P = torch.exp(log_P)

        # Check row sums
        row_sum = P.sum(dim=-1)
        row_err = (row_sum - a).abs().max()
        assert row_err < 1e-4, f"Row marginal error too large: {row_err}"

        # Check column sums
        col_sum = P.sum(dim=-2)
        col_err = (col_sum - b).abs().max()
        assert col_err < 1e-4, f"Column marginal error too large: {col_err}"

    def test_transport_plan_nonnegative(self, simple_problem):
        """Test that transport plan is non-negative."""
        C, a, b = simple_problem
        epsilon = 0.1

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=epsilon)

        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon
        P = torch.exp(log_P)

        assert (P >= 0).all(), "Transport plan has negative entries"

    def test_uniform_marginals_default(self, device):
        """Test with default uniform marginals."""
        B, N, M = 4, 16, 16
        C = torch.randn(B, N, M, device=device).abs()

        f, g, _, converged = sinkhorn_balanced(C, a=None, b=None, epsilon=0.1)

        assert converged
        assert f.shape == (B, N)
        assert g.shape == (B, M)

    def test_small_epsilon_stability(self, device):
        """Test numerical stability with small epsilon."""
        B, N, M = 2, 32, 32
        C = torch.randn(B, N, M, device=device).abs() * 10
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        # Small epsilon - should not produce NaN
        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=0.01, max_iters=500)

        assert not torch.isnan(f).any(), "f contains NaN"
        assert not torch.isnan(g).any(), "g contains NaN"
        assert not torch.isinf(f).any(), "f contains Inf"
        assert not torch.isinf(g).any(), "g contains Inf"

    def test_large_epsilon_smoothness(self, device):
        """Test that large epsilon gives smooth solution."""
        B, N, M = 2, 16, 16
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.ones(B, N, device=device) / N
        b = torch.ones(B, M, device=device) / M

        # Large epsilon - should converge quickly
        f, g, n_iters, converged = sinkhorn_balanced(
            C, a, b, epsilon=1.0, max_iters=50, threshold=1e-6
        )

        assert converged
        assert n_iters < 30, f"Large epsilon should converge fast, got {n_iters}"

    def test_batch_independence(self, device):
        """Test that batch elements are computed independently."""
        B, N, M = 4, 16, 16
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        # Full batch
        f_full, g_full, _, _ = sinkhorn_balanced(C, a, b, epsilon=0.1)

        # Single batch elements
        for i in range(B):
            f_i, g_i, _, _ = sinkhorn_balanced(
                C[i : i + 1], a[i : i + 1], b[i : i + 1], epsilon=0.1
            )

            f_err = (f_full[i] - f_i[0]).abs().max()
            g_err = (g_full[i] - g_i[0]).abs().max()

            assert f_err < 1e-5, f"Batch {i} f differs: {f_err}"
            assert g_err < 1e-5, f"Batch {i} g differs: {g_err}"


class TestNumericalPrecision:
    """Tests for numerical precision and edge cases."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_extreme_cost_values(self, device):
        """Test with very large cost values."""
        B, N, M = 2, 16, 16
        C = torch.randn(B, N, M, device=device).abs() * 100  # Large costs
        a = torch.ones(B, N, device=device) / N
        b = torch.ones(B, M, device=device) / M

        f, g, _, converged = sinkhorn_balanced(C, a, b, epsilon=1.0, max_iters=200)

        assert not torch.isnan(f).any(), "NaN with large costs"
        assert not torch.isinf(f).any(), "Inf with large costs"

    def test_sparse_marginals(self, device):
        """Test with near-sparse marginals."""
        B, N, M = 2, 16, 16

        # Sparse marginals (some very small values)
        a = torch.zeros(B, N, device=device)
        a[:, :4] = 0.25  # Only first 4 have mass

        b = torch.zeros(B, M, device=device)
        b[:, :4] = 0.25

        C = torch.randn(B, N, M, device=device).abs()

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=0.1)

        assert not torch.isnan(f).any()
        assert not torch.isnan(g).any()

    def test_transport_plan_sums_to_one(self, device):
        """Test that transport plan sums to 1."""
        B, N, M = 4, 32, 32
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=0.1, max_iters=200)

        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / 0.1
        P = torch.exp(log_P)
        total_mass = P.sum(dim=(-2, -1))

        err = (total_mass - 1.0).abs().max()
        assert err < 1e-4, f"Total mass should be 1, error: {err}"

    def test_lse_stability(self, device):
        """Test that log-sum-exp is computed stably."""
        B, N, M = 2, 64, 64

        # Create cost matrix with large range
        C = torch.zeros(B, N, M, device=device)
        C[:, 0, 0] = -100  # Very favorable
        C[:, -1, -1] = 100  # Very unfavorable

        a = torch.ones(B, N, device=device) / N
        b = torch.ones(B, M, device=device) / M

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=1.0, max_iters=100)

        assert not torch.isnan(f).any(), "LSE should handle large range"
        assert not torch.isinf(f).any(), "LSE should not overflow"
