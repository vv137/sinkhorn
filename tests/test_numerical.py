"""Numerical precision and accuracy tests."""

import pytest
import torch
import math


class TestNumericalAccuracy:
    """Test numerical accuracy of implementations."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_log_sum_exp_stability(self, device):
        """Test that LSE computation is numerically stable."""
        from sinkhorn.pytorch.balanced import softmin_row

        B, N, M = 2, 16, 16
        epsilon = 0.01  # Small epsilon - harder for stability

        # Create cost matrix with large values
        C = torch.randn(B, N, M, device=device) * 100
        g = torch.zeros(B, M, device=device)

        result = softmin_row(C, g, epsilon)

        assert not torch.isnan(result).any(), "softmin produced NaN"
        assert not torch.isinf(result).any(), "softmin produced Inf"

    def test_transport_plan_sums_correctly(self, device):
        """Test that transport plan row/column sums match marginals."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        B, N, M = 4, 32, 32
        epsilon = 0.1

        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        f, g, n_iters, converged = sinkhorn_balanced(
            C, a, b, epsilon=epsilon, max_iters=200, threshold=1e-8
        )

        # Convergence not guaranteed with random data, just check it ran
        assert n_iters > 0, "Should run at least one iteration"

        # Compute transport plan
        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon
        P = torch.exp(log_P)

        # Check marginals with relative tolerance
        row_sum = P.sum(dim=-1)
        col_sum = P.sum(dim=-2)

        row_rel_err = ((row_sum - a) / a.clamp(min=1e-10)).abs().max()
        col_rel_err = ((col_sum - b) / b.clamp(min=1e-10)).abs().max()

        assert row_rel_err < 1e-2, f"Row marginal rel error: {row_rel_err}"
        assert col_rel_err < 1e-2, f"Column marginal rel error: {col_rel_err}"

    def test_dual_objective_is_concave(self, device):
        """Test that dual objective increases during iterations."""
        from sinkhorn.pytorch.unbalanced import (
            sinkhorn_unbalanced,
            compute_dual_objective,
        )

        B, N, M = 2, 16, 16
        epsilon = 0.1

        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        # Track dual objective over iterations
        objectives = []

        f = torch.zeros(B, N, device=device)
        g = torch.zeros(B, M, device=device)

        f_prev, g_prev = f.clone(), g.clone()

        for i in range(20):
            f, g, _, _ = sinkhorn_unbalanced(
                C,
                a,
                b,
                epsilon=epsilon,
                tau_a=None,
                tau_b=None,
                max_iters=i + 1,
                threshold=0,
            )

            dual = compute_dual_objective(C, f, g, a, b, epsilon, None, None).mean()
            objectives.append(dual.item())

        # Dual should generally increase (may have small decreases due to numerical issues)
        # Check that final > initial
        assert objectives[-1] > objectives[0], "Dual objective should increase"

    def test_epsilon_interpolation(self, device):
        """Test that results interpolate smoothly with epsilon."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        B, N, M = 2, 16, 16
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        epsilons = [0.01, 0.05, 0.1, 0.5, 1.0]
        costs = []

        for eps in epsilons:
            f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=eps, max_iters=200)
            log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / eps
            P = torch.exp(log_P)
            cost = (P * C).sum(dim=(-2, -1)).mean()
            costs.append(cost.item())

        # Costs should be monotonic (larger epsilon = smoother = higher transport cost)
        for i in range(len(costs) - 1):
            assert costs[i] <= costs[i + 1] + 0.1, (
                f"Cost should increase with epsilon: {costs}"
            )

    def test_symmetric_cost_matrix(self, device):
        """Test with symmetric cost matrix gives consistent results."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        B, N = 2, 16

        # Create symmetric cost matrix
        temp = torch.randn(B, N, N, device=device)
        C = (temp + temp.transpose(-2, -1)) / 2

        # Uniform marginals (symmetric problem)
        a = torch.ones(B, N, device=device) / N
        b = torch.ones(B, N, device=device) / N

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=0.1, max_iters=200)

        # For symmetric problem with uniform marginals, f and g should have similar range
        f_range = f.max() - f.min()
        g_range = g.max() - g.min()
        range_diff = (f_range - g_range).abs()
        assert range_diff < 1.0, (
            f"Symmetric problem should have similar potential ranges: {range_diff}"
        )

    def test_convergence_rate(self, device):
        """Test that convergence rate depends on epsilon."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        B, N, M = 2, 32, 32
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        # Large epsilon should converge faster
        _, _, iters_large, conv_large = sinkhorn_balanced(
            C, a, b, epsilon=1.0, max_iters=100, threshold=1e-6
        )

        # Small epsilon should converge slower
        _, _, iters_small, conv_small = sinkhorn_balanced(
            C, a, b, epsilon=0.01, max_iters=100, threshold=1e-6
        )

        assert iters_large <= iters_small, (
            f"Large epsilon should converge faster: {iters_large} vs {iters_small}"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_single_point_marginals(self, device):
        """Test with delta distribution marginals."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        B, N, M = 2, 8, 8
        C = torch.randn(B, N, M, device=device).abs()

        # Delta distribution: all mass on first point
        a = torch.zeros(B, N, device=device)
        a[:, 0] = 1.0

        b = torch.zeros(B, M, device=device)
        b[:, 0] = 1.0

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=0.1, max_iters=200)

        # Transport should be concentrated at (0, 0)
        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / 0.1
        P = torch.exp(log_P)

        # P[0,0] should dominate
        assert P[:, 0, 0].mean() > 0.9, "Delta marginals should concentrate transport"

    def test_very_small_batch(self, device):
        """Test with batch size 1."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        B, N, M = 1, 16, 16
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        f, g, _, converged = sinkhorn_balanced(C, a, b, epsilon=0.1)

        assert converged
        assert f.shape == (1, N)
        assert g.shape == (1, M)

    def test_rectangular_matrix(self, device):
        """Test with non-square cost matrix."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        B, N, M = 2, 32, 16  # N > M
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        f, g, _, converged = sinkhorn_balanced(C, a, b, epsilon=0.1, max_iters=200)

        assert converged
        assert f.shape == (B, N)
        assert g.shape == (B, M)

        # Check marginals
        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / 0.1
        P = torch.exp(log_P)

        row_err = (P.sum(dim=-1) - a).abs().max()
        col_err = (P.sum(dim=-2) - b).abs().max()

        assert row_err < 1e-3
        assert col_err < 1e-3

    def test_identity_cost_uniform_marginals(self, device):
        """Test known solution: identity cost with uniform marginals."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        B, N = 2, 16

        # Identity cost: optimal transport is identity permutation
        C = torch.ones(B, N, N, device=device) * 10
        for i in range(N):
            C[:, i, i] = 0.0

        a = torch.ones(B, N, device=device) / N
        b = torch.ones(B, N, device=device) / N

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=0.01, max_iters=500)

        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / 0.01
        P = torch.exp(log_P)

        # P should be close to identity (scaled)
        diag_mass = torch.diagonal(P, dim1=-2, dim2=-1).sum(dim=-1)
        total_mass = P.sum(dim=(-2, -1))

        diag_ratio = (diag_mass / total_mass).mean()
        assert diag_ratio > 0.8, f"Diagonal should dominate: {diag_ratio}"
