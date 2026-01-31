# Sinkhorn-Triton

High-performance Triton kernels for balanced and unbalanced Sinkhorn optimal transport.

## Features

- 🚀 **Triton-accelerated** kernels with O(N+M) memory via streaming LSE
- ⚖️ **Unified API** for both balanced and unbalanced optimal transport
- 🎭 **Masking support** for variable-length sequences
- 🔄 **Implicit differentiation** for memory-efficient backward pass
- 📊 **Pure PyTorch fallback** for comparison and debugging

## Installation

```bash
# Clone the repository
git clone https://github.com/vv137/sinkhorn-triton.git
cd sinkhorn-triton

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Triton 2.0+ (for GPU kernels)
- CUDA-capable GPU (optional, for Triton backend)

## Quick Start

```python
import torch
from sinkhorn import sinkhorn

# Cost matrix and marginals
C = torch.randn(8, 64, 64, device='cuda')
a = torch.softmax(torch.randn(8, 64, device='cuda'), dim=-1)
b = torch.softmax(torch.randn(8, 64, device='cuda'), dim=-1)

# Balanced Sinkhorn
out = sinkhorn(C, a, b, epsilon=0.1)
print(f"Converged: {out.converged}, Iterations: {out.n_iters}")

# Unbalanced Sinkhorn (with KL relaxation)
out = sinkhorn(C, a, b, epsilon=0.1, tau_a=1.0, tau_b=1.0)

# Get transport plan (computed on-the-fly)
P = out.transport_plan(C, epsilon=0.1)
```

## Benchmark

| Library | Backend | Time | Speedup |
| ------- | ------- | ---- | ------- |
| **sinkhorn-triton** | **Triton** | **2.35ms** | **21x** vs POT-GPU |
| sinkhorn-triton | PyTorch | 4.65ms | 11x |
| POT | NumPy | 6.22ms | 8x |
| POT | CUDA | 49.12ms | 1x |

```bash
uv run python scripts/benchmark.py --sizes 32 64 128 256 --batch-sizes 1 4 8
```

## Architecture

```
src/sinkhorn/
├── sinkhorn.py        # High-level API entry point
├── pytorch/           # Pure PyTorch reference implementations
│   ├── balanced.py    # Balanced Sinkhorn (τ=∞)
│   ├── unbalanced.py  # Unbalanced Sinkhorn with KL relaxation
│   └── implicit_diff.py
├── triton/            # Triton kernel implementations
│   ├── lse_kernel.py  # Streaming log-sum-exp
│   ├── forward.py     # Row/col update kernels
│   └── jvp_kernel.py  # JVP for implicit differentiation
└── autograd.py        # torch.autograd.Function wrappers
```

## Mathematical Background

### Unified Update Rule

Balanced and Unbalanced OT unified via scaling factor ρ:

- **Unbalanced**: `ρ = ε / (ε + τ)`
- **Balanced**: `ρ = 0` (τ → ∞)

Update equations (log-space):

```
f ← ρ·f + (1-ρ)·(-ε·logsumexp((g - C)/ε, dim=M) + ε·log(a))
g ← ρ·g + (1-ρ)·(-ε·logsumexp((f - C)/ε, dim=N) + ε·log(b))
```

### Implicit Differentiation

Backward pass uses Neumann series to solve adjoint system without storing intermediate iterations.

## API Reference

### `sinkhorn(C, a, b, epsilon, ...)`

**Parameters:**

- `C`: Cost matrix `(B, N, M)`
- `a`: Source marginal `(B, N)` (default: uniform)
- `b`: Target marginal `(B, M)` (default: uniform)
- `epsilon`: Entropic regularization coefficient
- `tau_a`, `tau_b`: KL relaxation parameters (None = balanced)
- `mask_a`, `mask_b`: Boolean masks for valid entries
- `backend`: `"triton"` or `"pytorch"`
- `max_iters`: Maximum iterations (default: 100)
- `threshold`: Convergence threshold (default: 1e-6)

**Returns:** `SinkhornOutput` with dual potentials `f`, `g`

## Testing

```bash
# All tests
uv run pytest tests/ -v

# Gradient tests
uv run pytest tests/test_gradients.py -v

# Benchmark tests
uv run pytest tests/test_benchmark.py --benchmark-only
```

## License

MIT
