# Sinkhorn

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
git clone https://github.com/vv137/sinkhorn.git
cd sinkhorn

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Requirements

- Python 3.11+
- PyTorch 2.0+
- Triton 2.1+ (for GPU kernels)
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

Performance comparison on **NVIDIA RTX 6000 Ada Generation** (N=M=[128...16384], Batch=1):

| Library           | Algorithm            | Avg Time (16k) | Characteristics |
| ----------------- | -------------------- | -------------- | ------------------ |
| **sinkhorn**      | **Stabilized (Log)** | **739ms**      | **~3x faster than POT (Log)**, numerically stable |
| POT               | Stabilized (Log)     | 2181ms         | Robust implementation via `method='sinkhorn_log'` |
| POT               | Standard (Scaling)   | 278ms          | Fastest but numerically unstable (NaN risk with small ε) |
| sinkhorn          | Stabilized (Log)     | 2618ms         | Pure PyTorch fallback |

```bash
uv run python scripts/benchmark.py --batch-sizes 1 --sizes 128 256 512 1024 2048 4096 8192 12288 16384
```

## Architecture

```text
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

```text
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
