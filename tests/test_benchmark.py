"""Pytest-based benchmark tests."""

import pytest
import torch

# Check if pytest-benchmark is available
try:
    import pytest_benchmark

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
class TestBenchmarkBalanced:
    """Benchmark balanced Sinkhorn."""

    @pytest.mark.parametrize("N", [32, 64, 128])
    @pytest.mark.parametrize("backend", ["pytorch", "triton"])
    def test_benchmark_size(self, benchmark, device, N, backend):
        """Benchmark different matrix sizes."""
        if backend == "triton" and not torch.cuda.is_available():
            pytest.skip("Triton requires CUDA")

        from sinkhorn import sinkhorn

        B = 8
        C = torch.randn(B, N, N, device=device)
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, N, device=device), dim=-1)

        def run():
            out = sinkhorn(C, a, b, epsilon=0.1, max_iters=100, backend=backend)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return out

        result = benchmark(run)
        # Check no NaN (convergence may vary based on random input)
        assert not torch.isnan(result.f).any(), "f contains NaN"
        assert not torch.isnan(result.g).any(), "g contains NaN"


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
class TestBenchmarkUnbalanced:
    """Benchmark unbalanced Sinkhorn."""

    @pytest.mark.parametrize("tau", [0.5, 1.0, 5.0])
    def test_benchmark_tau(self, benchmark, device, tau):
        """Benchmark different τ values."""
        from sinkhorn import sinkhorn

        B, N = 8, 64
        C = torch.randn(B, N, N, device=device)
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, N, device=device), dim=-1)

        def run():
            out = sinkhorn(
                C,
                a,
                b,
                epsilon=0.1,
                tau_a=tau,
                tau_b=tau,
                max_iters=50,
                backend="pytorch",
            )
            return out

        result = benchmark(run)


class TestCompareBackends:
    """Compare PyTorch vs Triton results."""

    @pytest.fixture
    def problem(self, device):
        """Create test problem."""
        B, N, M = 4, 32, 32
        C = torch.randn(B, N, M, device=device)
        a = torch.softmax(torch.randn(B, N, device=device), dim=-1)
        b = torch.softmax(torch.randn(B, M, device=device), dim=-1)
        return C, a, b

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_backends_agree(self, problem):
        """Test that PyTorch and Triton give same results."""
        from sinkhorn import sinkhorn

        C, a, b = problem
        epsilon = 0.1
        max_iters = 100

        out_pytorch = sinkhorn(
            C, a, b, epsilon=epsilon, max_iters=max_iters, backend="pytorch"
        )
        out_triton = sinkhorn(
            C, a, b, epsilon=epsilon, max_iters=max_iters, backend="triton"
        )

        # Potentials should be close (Triton uses different numerical path)
        f_err = (out_pytorch.f - out_triton.f).abs().max()
        g_err = (out_pytorch.g - out_triton.g).abs().max()

        assert f_err < 0.5, f"f differs too much: {f_err}"
        assert g_err < 0.5, f"g differs too much: {g_err}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_transport_plans_agree(self, problem):
        """Test that transport plans from both backends agree."""
        from sinkhorn import sinkhorn

        C, a, b = problem
        epsilon = 0.1

        out_pytorch = sinkhorn(C, a, b, epsilon=epsilon, backend="pytorch")
        out_triton = sinkhorn(C, a, b, epsilon=epsilon, backend="triton")

        P_pytorch = out_pytorch.transport_plan(C, epsilon)
        P_triton = out_triton.transport_plan(C, epsilon)

        # Transport plans should match within tolerance
        err = (P_pytorch - P_triton).abs().max()
        assert err < 0.1, f"Transport plans differ too much: {err}"
