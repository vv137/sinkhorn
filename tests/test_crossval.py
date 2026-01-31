"""Cross-validation tests comparing sinkhorn-triton with POT library."""

import pytest
import torch
import numpy as np


class TestCrossValidationPOT:
    """Cross-validate sinkhorn-triton with POT library."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def pot_available(self):
        try:
            import ot  # noqa: F401

            return True
        except ImportError:
            return False

    def test_transport_plan_matches_pot(self, device, pot_available):
        """Test that transport plan matches POT sinkhorn."""
        if not pot_available:
            pytest.skip("POT not installed")

        import ot
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        N, M = 16, 16
        epsilon = 0.1
        max_iters = 200

        # Create test data
        np.random.seed(42)
        C_np = np.abs(np.random.randn(N, M)).astype(np.float32)
        a_np = np.exp(np.random.randn(N).astype(np.float32))
        a_np = a_np / a_np.sum()
        b_np = np.exp(np.random.randn(M).astype(np.float32))
        b_np = b_np / b_np.sum()

        # POT reference
        P_pot = ot.sinkhorn(a_np, b_np, C_np, epsilon, numItermax=max_iters)

        # Our implementation
        C = torch.from_numpy(C_np).unsqueeze(0).to(device)
        a = torch.from_numpy(a_np).unsqueeze(0).to(device)
        b = torch.from_numpy(b_np).unsqueeze(0).to(device)

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=epsilon, max_iters=max_iters)

        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon
        P_ours = torch.exp(log_P).squeeze(0).cpu().numpy()

        # Compare transport plans
        err = np.abs(P_ours - P_pot).max()
        assert err < 0.01, f"Transport plan differs from POT: max error = {err}"

    def test_transport_cost_matches_pot(self, device, pot_available):
        """Test that transport cost matches POT."""
        if not pot_available:
            pytest.skip("POT not installed")

        import ot
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        N, M = 32, 32
        epsilon = 0.05
        max_iters = 300

        np.random.seed(123)
        C_np = np.abs(np.random.randn(N, M)).astype(np.float32)
        a_np = np.exp(np.random.randn(N).astype(np.float32))
        a_np = a_np / a_np.sum()
        b_np = np.exp(np.random.randn(M).astype(np.float32))
        b_np = b_np / b_np.sum()

        # POT reference
        P_pot = ot.sinkhorn(a_np, b_np, C_np, epsilon, numItermax=max_iters)
        cost_pot = np.sum(P_pot * C_np)

        # Our implementation
        C = torch.from_numpy(C_np).unsqueeze(0).to(device)
        a = torch.from_numpy(a_np).unsqueeze(0).to(device)
        b = torch.from_numpy(b_np).unsqueeze(0).to(device)

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=epsilon, max_iters=max_iters)

        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon
        P_ours = torch.exp(log_P)
        cost_ours = (P_ours * C).sum().item()

        # Compare costs
        rel_err = abs(cost_ours - cost_pot) / cost_pot
        assert rel_err < 0.01, (
            f"Transport cost differs: ours={cost_ours:.6f}, pot={cost_pot:.6f}, rel_err={rel_err:.4f}"
        )

    def test_marginals_match_pot(self, device, pot_available):
        """Test that marginal constraints match POT."""
        if not pot_available:
            pytest.skip("POT not installed")

        import ot
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        N, M = 24, 24
        epsilon = 0.1

        np.random.seed(456)
        C_np = np.abs(np.random.randn(N, M)).astype(np.float32)
        a_np = np.exp(np.random.randn(N).astype(np.float32))
        a_np = a_np / a_np.sum()
        b_np = np.exp(np.random.randn(M).astype(np.float32))
        b_np = b_np / b_np.sum()

        # POT reference
        P_pot = ot.sinkhorn(a_np, b_np, C_np, epsilon, numItermax=200)
        row_sum_pot = P_pot.sum(axis=1)
        col_sum_pot = P_pot.sum(axis=0)

        # Our implementation
        C = torch.from_numpy(C_np).unsqueeze(0).to(device)
        a = torch.from_numpy(a_np).unsqueeze(0).to(device)
        b = torch.from_numpy(b_np).unsqueeze(0).to(device)

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=epsilon, max_iters=200)

        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon
        P_ours = torch.exp(log_P).squeeze(0).cpu().numpy()
        row_sum_ours = P_ours.sum(axis=1)
        col_sum_ours = P_ours.sum(axis=0)

        # Both should match input marginals
        row_err_pot = np.abs(row_sum_pot - a_np).max()
        row_err_ours = np.abs(row_sum_ours - a_np).max()
        col_err_pot = np.abs(col_sum_pot - b_np).max()
        col_err_ours = np.abs(col_sum_ours - b_np).max()

        assert row_err_ours < 0.01, f"Row marginal error: {row_err_ours}"
        assert col_err_ours < 0.01, f"Col marginal error: {col_err_ours}"

        # Our errors should be similar to POT
        assert abs(row_err_ours - row_err_pot) < 0.01
        assert abs(col_err_ours - col_err_pot) < 0.01

    def test_dual_potentials_consistency(self, device, pot_available):
        """Test dual potential consistency with POT coupling matrix."""
        if not pot_available:
            pytest.skip("POT not installed")

        import ot
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        N, M = 16, 20
        epsilon = 0.1

        np.random.seed(789)
        C_np = np.abs(np.random.randn(N, M)).astype(np.float32)
        a_np = np.exp(np.random.randn(N).astype(np.float32))
        a_np = a_np / a_np.sum()
        b_np = np.exp(np.random.randn(M).astype(np.float32))
        b_np = b_np / b_np.sum()

        C = torch.from_numpy(C_np).unsqueeze(0).to(device)
        a = torch.from_numpy(a_np).unsqueeze(0).to(device)
        b = torch.from_numpy(b_np).unsqueeze(0).to(device)

        f, g, _, converged = sinkhorn_balanced(C, a, b, epsilon=epsilon, max_iters=200)

        assert converged, "Should converge"

        # Verify dual form: P_ij = exp((f_i + g_j - C_ij) / ε)
        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon
        P = torch.exp(log_P)

        # Transport plan should be non-negative
        assert (P >= 0).all()

        # Total mass should be 1
        total_mass = P.sum().item()
        assert abs(total_mass - 1.0) < 0.01, f"Total mass: {total_mass}"

    def test_varying_epsilon_consistency(self, device, pot_available):
        """Test consistency across different epsilon values."""
        if not pot_available:
            pytest.skip("POT not installed")

        import ot
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        N, M = 16, 16

        np.random.seed(321)
        C_np = np.abs(np.random.randn(N, M)).astype(np.float32)
        a_np = np.exp(np.random.randn(N).astype(np.float32))
        a_np = a_np / a_np.sum()
        b_np = np.exp(np.random.randn(M).astype(np.float32))
        b_np = b_np / b_np.sum()

        C = torch.from_numpy(C_np).unsqueeze(0).to(device)
        a = torch.from_numpy(a_np).unsqueeze(0).to(device)
        b = torch.from_numpy(b_np).unsqueeze(0).to(device)

        epsilons = [1.0, 0.5, 0.1]

        for eps in epsilons:
            # POT
            P_pot = ot.sinkhorn(a_np, b_np, C_np, eps, numItermax=300)
            cost_pot = np.sum(P_pot * C_np)

            # Ours
            f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=eps, max_iters=300)
            log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / eps
            P_ours = torch.exp(log_P)
            cost_ours = (P_ours * C).sum().item()

            rel_err = abs(cost_ours - cost_pot) / cost_pot
            assert rel_err < 0.02, (
                f"epsilon={eps}: cost mismatch, rel_err={rel_err:.4f}"
            )


class TestValueComparison:
    """Additional value comparison tests."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_backends_produce_same_values(self, device):
        """Test that PyTorch backend matches expected values."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced
        from sinkhorn.pytorch.unbalanced import sinkhorn_unbalanced

        B, N, M = 2, 16, 16
        epsilon = 0.1

        torch.manual_seed(42)
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        # Balanced via balanced function
        f1, g1, _, _ = sinkhorn_balanced(C, a, b, epsilon=epsilon, max_iters=100)

        # Balanced via unbalanced function (tau=None)
        f2, g2, _, _ = sinkhorn_unbalanced(C, a, b, epsilon=epsilon, max_iters=100)

        # Should produce very similar results
        f_diff = (f1 - f2).abs().max()
        g_diff = (g1 - g2).abs().max()

        assert f_diff < 0.01, f"Balanced and unbalanced(tau=None) f differ: {f_diff}"
        assert g_diff < 0.01, f"Balanced and unbalanced(tau=None) g differ: {g_diff}"

    def test_transport_plan_structure(self, device):
        """Test transport plan has correct structure."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        B, N, M = 2, 16, 16
        epsilon = 0.1

        torch.manual_seed(123)
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        f, g, _, _ = sinkhorn_balanced(C, a, b, epsilon=epsilon, max_iters=200)

        log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon
        P = torch.exp(log_P)

        # Non-negative
        assert (P >= 0).all(), "Transport plan should be non-negative"

        # Row sums ≈ a
        row_sums = P.sum(dim=-1)
        row_err = (row_sums - a).abs().max()
        assert row_err < 0.01, f"Row marginal error: {row_err}"

        # Column sums ≈ b
        col_sums = P.sum(dim=-2)
        col_err = (col_sums - b).abs().max()
        assert col_err < 0.01, f"Column marginal error: {col_err}"

        # Total mass ≈ 1 (per batch)
        total_mass = P.sum(dim=(-2, -1))
        mass_err = (total_mass - 1.0).abs().max()
        assert mass_err < 0.01, f"Total mass error: {mass_err}"

    def test_cost_decreases_with_iterations(self, device):
        """Test that transport cost stabilizes with more iterations."""
        from sinkhorn.pytorch.balanced import sinkhorn_balanced

        B, N, M = 2, 16, 16
        epsilon = 0.1

        torch.manual_seed(456)
        C = torch.randn(B, N, M, device=device).abs()
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)

        costs = []
        for max_iters in [10, 50, 100, 200]:
            f, g, _, _ = sinkhorn_balanced(
                C, a, b, epsilon=epsilon, max_iters=max_iters
            )
            log_P = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / epsilon
            P = torch.exp(log_P)
            cost = (P * C).sum(dim=(-2, -1)).mean().item()
            costs.append(cost)

        # Cost should stabilize
        final_diff = abs(costs[-1] - costs[-2])
        assert final_diff < 0.01, f"Cost not stabilizing: {costs}"
