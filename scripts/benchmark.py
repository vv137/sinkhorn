"""Benchmark Triton vs PyTorch vs POT implementations with SVG plot output."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def benchmark_sinkhorn_internal(
    backend: str,
    B: int,
    N: int,
    M: int,
    epsilon: float = 1.0,
    max_iters: int = 100,
    warmup: int = 5,
    repeats: int = 10,
    device: str = "cuda",
) -> dict[str, Any]:
    """Benchmark sinkhorn backends.

    Returns:
        Dictionary with timing results
    """
    from sinkhorn import sinkhorn

    torch_device = torch.device(device)

    # Create data
    C = torch.randn(B, N, M, device=torch_device).abs()
    a = torch.softmax(torch.randn(B, N, device=torch_device), dim=-1)
    b = torch.softmax(torch.randn(B, M, device=torch_device), dim=-1)

    # Warmup
    for _ in range(warmup):
        out = sinkhorn(C, a, b, epsilon=epsilon, max_iters=max_iters, backend=backend)
        if device == "cuda":
            torch.cuda.synchronize()

    # Reset memory stats before benchmarking
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        out = sinkhorn(C, a, b, epsilon=epsilon, max_iters=max_iters, backend=backend)

        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    # Get peak memory usage
    peak_memory_mb = 0.0
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "library": "sinkhorn",
        "backend": backend,
        "B": B,
        "N": N,
        "M": M,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times))
        ** 0.5,
        "peak_memory_mb": peak_memory_mb,
    }


def benchmark_pot(
    B: int,
    N: int,
    M: int,
    epsilon: float = 0.1,
    max_iters: int = 100,
    warmup: int = 3,
    repeats: int = 10,
    device: str = "cuda",
    method: str = "sinkhorn",
) -> dict[str, Any]:
    """Benchmark POT (Python Optimal Transport) library.

    Returns:
        Dictionary with timing results
    """
    try:
        import ot
    except ImportError:
        return {"library": f"pot-{method}", "error": "POT not installed"}

    torch_device = torch.device(device)

    # POT works on numpy/single samples, so we run B iterations
    times = []

    # Create data (single sample for POT)
    C_np = np.abs(np.random.randn(N, M).astype(np.float32))
    a_np = np.exp(np.random.randn(N).astype(np.float32))
    a_np = a_np / a_np.sum()
    b_np = np.exp(np.random.randn(M).astype(np.float32))
    b_np = b_np / b_np.sum()

    # Warmup
    for _ in range(warmup):
        _ = ot.sinkhorn(a_np, b_np, C_np, epsilon, numItermax=max_iters, method=method)

    # Benchmark (scale by batch size for fair comparison)
    for _ in range(repeats):
        start = time.perf_counter()

        for _ in range(B):
            _ = ot.sinkhorn(
                a_np, b_np, C_np, epsilon, numItermax=max_iters, method=method
            )

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "library": f"pot-{method}",
        "backend": "numpy",
        "B": B,
        "N": N,
        "M": M,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times))
        ** 0.5,
    }


def benchmark_pot_gpu(
    B: int,
    N: int,
    M: int,
    epsilon: float = 0.1,
    max_iters: int = 100,
    warmup: int = 3,
    repeats: int = 10,
    device: str = "cuda",
    method: str = "sinkhorn",
) -> dict[str, Any]:
    """Benchmark POT with PyTorch backend (GPU).

    Returns:
        Dictionary with timing results
    """
    try:
        import ot
    except ImportError:
        return {"library": f"pot-gpu-{method}", "error": "POT not installed"}

    if device != "cuda" or not torch.cuda.is_available():
        return {"library": f"pot-gpu-{method}", "error": "CUDA not available"}

    torch_device = torch.device(device)

    # Create data on GPU (POT supports torch tensors)
    C = torch.randn(B, N, M, device=torch_device).abs()
    a = torch.softmax(torch.randn(B, N, device=torch_device), dim=-1)
    b = torch.softmax(torch.randn(B, M, device=torch_device), dim=-1)

    times = []

    # Warmup
    for _ in range(warmup):
        for i in range(B):
            _ = ot.sinkhorn(
                a[i], b[i], C[i], epsilon, numItermax=max_iters, method=method
            )
        torch.cuda.synchronize()

    # Reset memory stats before benchmarking
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()

        for i in range(B):
            _ = ot.sinkhorn(
                a[i], b[i], C[i], epsilon, numItermax=max_iters, method=method
            )

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Get peak memory usage
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "library": f"pot-gpu-{method}",
        "backend": "torch-cuda",
        "B": B,
        "N": N,
        "M": M,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times))
        ** 0.5,
        "peak_memory_mb": peak_memory_mb,
    }


def run_benchmark_suite(
    sizes: list[int],
    batch_sizes: list[int],
    output_dir: Path,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Run full benchmark suite.

    Args:
        sizes: List of N=M sizes to test
        batch_sizes: List of batch sizes to test
        output_dir: Directory to save results
        device: Device to run on

    Returns:
        List of benchmark results
    """
    results = []

    # Determine backends to test
    internal_backends = ["pytorch"]
    if device == "cuda":
        internal_backends.append("triton")

    total_configs = len(sizes) * len(batch_sizes)
    current = 0

    for N in sizes:
        for B in batch_sizes:
            current += 1
            print(f"\n[{current}/{total_configs}] Testing B={B} N={N}...")

            # Test internal backends
            for backend in internal_backends:
                try:
                    result = benchmark_sinkhorn_internal(
                        backend=backend,
                        B=B,
                        N=N,
                        M=N,
                        device=device,
                    )
                    results.append(result)
                    print(f"  sinkhorn ({backend}): {result['mean_ms']:.2f}ms")
                except Exception as e:
                    print(f"  sinkhorn ({backend}): ERROR - {e}")
                    results.append(
                        {
                            "library": "sinkhorn",
                            "backend": backend,
                            "B": B,
                            "N": N,
                            "M": N,
                            "error": str(e),
                        }
                    )

            # Test POT (CPU) - Standard
            if N < 4096:
                try:
                    result = benchmark_pot(
                        B=B, N=N, M=N, device=device, method="sinkhorn"
                    )
                    if "error" not in result:
                        results.append(result)
                        print(f"  POT (numpy-cpu): {result['mean_ms']:.2f}ms")
                    else:
                        print(f"  POT (numpy-cpu): {result['error']}")
                except Exception as e:
                    print(f"  POT (numpy-cpu): ERROR - {e}")
            else:
                print(f"  POT (numpy-cpu): Skipped (N={N} too large for CPU)")

            # Test POT (GPU via torch) - Standard
            if device == "cuda":
                try:
                    result = benchmark_pot_gpu(
                        B=B, N=N, M=N, device=device, method="sinkhorn"
                    )
                    if "error" not in result:
                        results.append(result)
                        print(f"  POT (torch-cuda): {result['mean_ms']:.2f}ms")
                    else:
                        print(f"  POT (torch-cuda): {result.get('error', 'unknown')}")
                except Exception as e:
                    print(f"  POT (torch-cuda): ERROR - {e}")

                # Test POT (GPU via torch) - Log-domain
                try:
                    result = benchmark_pot_gpu(
                        B=B, N=N, M=N, device=device, method="sinkhorn_log"
                    )
                    if "error" not in result:
                        results.append(result)
                        print(f"  POT-Log (torch-cuda): {result['mean_ms']:.2f}ms")
                    else:
                        print(
                            f"  POT-Log (torch-cuda): {result.get('error', 'unknown')}"
                        )
                except Exception as e:
                    print(f"  POT-Log (torch-cuda): ERROR - {e}")

    return results


def plot_benchmark_results(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate SVG plots from benchmark results.

    Args:
        results: List of benchmark results
        output_path: Path to save SVG file
    """
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        print("No valid results to plot")
        return

    # Get unique libraries and batch sizes
    libraries = list(
        set(f"{r['library']}({r.get('backend', 'default')})" for r in valid_results)
    )
    batch = max(set(r["B"] for r in valid_results))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Time vs Size ---
    ax1 = axes[0]

    colors = plt.cm.tab10.colors
    lib_colors = {lib: colors[i % len(colors)] for i, lib in enumerate(libraries)}

    for lib in libraries:
        lib_results = [
            r
            for r in valid_results
            if f"{r['library']}({r.get('backend', 'default')})" == lib
            and r["B"] == batch
        ]
        if not lib_results:
            continue

        sizes = sorted(set(r["N"] for r in lib_results))
        times = [
            next((r["mean_ms"] for r in lib_results if r["N"] == s), None)
            for s in sizes
        ]
        times = [t for t in times if t is not None]
        sizes = sizes[: len(times)]

        if sizes and times:
            ax1.plot(
                sizes,
                times,
                "o-",
                label=lib,
                linewidth=2,
                markersize=8,
                color=lib_colors[lib],
            )

    ax1.set_xlabel("Matrix Size (N=M)", fontsize=12)
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title(f"Sinkhorn Performance Comparison (Batch={batch})", fontsize=14)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")

    # --- Plot 2: Speedup vs Baseline ---
    ax2 = axes[1]

    # Determine baseline: prefer sinkhorn(pytorch) as it's always available
    baseline_lib = "sinkhorn(pytorch)"
    baseline_name = "PyTorch"

    # Check if baseline exists
    baseline_entries = [
        r
        for r in valid_results
        if f"{r['library']}({r.get('backend', 'default')})" == baseline_lib
        and r["B"] == batch
    ]

    if not baseline_entries:
        # Fallback to POT-GPU
        baseline_lib = "pot-gpu-sinkhorn(torch-cuda)"
        baseline_name = "POT-GPU"
        baseline_entries = [
            r
            for r in valid_results
            if f"{r['library']}({r.get('backend', 'default')})" == baseline_lib
            and r["B"] == batch
        ]

    if not baseline_entries:
        # Fallback to POT-CPU
        baseline_lib = "pot-sinkhorn(numpy)"
        baseline_name = "POT-CPU"
        baseline_entries = [
            r
            for r in valid_results
            if f"{r['library']}({r.get('backend', 'default')})" == baseline_lib
            and r["B"] == batch
        ]

    baseline_results = {r["N"]: r["mean_ms"] for r in baseline_entries}

    if baseline_results:
        sizes = sorted(baseline_results.keys())

        for lib in libraries:
            if lib == baseline_lib:
                continue

            lib_results = [
                r
                for r in valid_results
                if f"{r['library']}({r.get('backend', 'default')})" == lib
                and r["B"] == batch
            ]

            speedups = []
            speedup_sizes = []
            for s in sizes:
                # Find corresponding result for this library
                lib_time = next(
                    (r["mean_ms"] for r in lib_results if r["N"] == s), None
                )
                if lib_time:
                    speedups.append(baseline_results[s] / lib_time)
                    speedup_sizes.append(s)

            if speedups:
                ax2.plot(
                    speedup_sizes,
                    speedups,
                    "o-",
                    label=f"{lib} vs {baseline_name}",
                    linewidth=2,
                    markersize=8,
                    color=lib_colors[lib],
                )

        ax2.axhline(
            y=1,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"{baseline_name} baseline",
        )
        ax2.set_xlabel("Matrix Size (N=M)", fontsize=12)
        ax2.set_ylabel(f"Speedup vs {baseline_name}", fontsize=12)
        ax2.set_title(f"Speedup over {baseline_name} (Batch={batch})", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log", base=2)

    plt.tight_layout()

    # Save as SVG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path}")


def plot_batch_comparison(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot performance across different batch sizes.

    Args:
        results: List of benchmark results
        output_path: Path to save SVG file
    """
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    libraries = list(
        set(f"{r['library']}({r.get('backend', 'default')})" for r in valid_results)
    )
    sizes = sorted(set(r["N"] for r in valid_results))
    batch_sizes = sorted(set(r["B"] for r in valid_results))

    # Use largest size for batch comparison
    size = max(sizes)

    x = np.arange(len(batch_sizes))
    width = 0.8 / len(libraries)
    colors = plt.cm.tab10.colors

    for i, lib in enumerate(libraries):
        times = []
        for B in batch_sizes:
            lib_time = next(
                (
                    r["mean_ms"]
                    for r in valid_results
                    if f"{r['library']}({r.get('backend', 'default')})" == lib
                    and r["B"] == B
                    and r["N"] == size
                ),
                None,
            )
            times.append(lib_time if lib_time else 0)

        ax.bar(
            x + i * width - 0.4 + width / 2,
            times,
            width,
            label=lib,
            alpha=0.8,
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(f"Sinkhorn Performance vs Batch Size (N=M={size})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved batch comparison plot to {output_path}")


def plot_memory_usage(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot memory usage across different sizes.

    Args:
        results: List of benchmark results
        output_path: Path to save SVG file
    """
    # Filter results with memory information
    valid_results = [
        r for r in results if "error" not in r and r.get("peak_memory_mb", 0) > 0
    ]

    if not valid_results:
        print("No memory data available to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    libraries = list(
        set(f"{r['library']}({r.get('backend', 'default')})" for r in valid_results)
    )
    batch = max(set(r["B"] for r in valid_results))

    colors = plt.cm.tab10.colors
    lib_colors = {lib: colors[i % len(colors)] for i, lib in enumerate(libraries)}

    for lib in libraries:
        lib_results = [
            r
            for r in valid_results
            if f"{r['library']}({r.get('backend', 'default')})" == lib
            and r["B"] == batch
        ]
        if not lib_results:
            continue

        sizes = sorted(set(r["N"] for r in lib_results))
        memory = [
            next((r["peak_memory_mb"] for r in lib_results if r["N"] == s), None)
            for s in sizes
        ]
        memory = [m for m in memory if m is not None]
        sizes = sizes[: len(memory)]

        if sizes and memory:
            ax.plot(
                sizes,
                memory,
                "o-",
                label=lib,
                linewidth=2,
                markersize=8,
                color=lib_colors[lib],
            )

    ax.set_xlabel("Matrix Size (N=M)", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_title(f"GPU Memory Usage (Batch={batch})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved memory usage plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Sinkhorn implementations")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SINKHORN BENCHMARK SUITE")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Sizes: {args.sizes}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Libraries: sinkhorn (pytorch, triton), POT (numpy, torch)")

    # Run benchmarks
    results = run_benchmark_suite(
        sizes=args.sizes,
        batch_sizes=args.batch_sizes,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Save raw results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {results_path}")

    # Generate plots
    plot_benchmark_results(results, args.output_dir / "performance.svg")
    plot_batch_comparison(results, args.output_dir / "batch_comparison.svg")
    plot_memory_usage(results, args.output_dir / "memory_usage.svg")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    valid_results = [r for r in results if "error" not in r]
    libraries = sorted(
        set(f"{r['library']}({r.get('backend', 'default')})" for r in valid_results)
    )

    for lib in libraries:
        lib_results = [
            r
            for r in valid_results
            if f"{r['library']}({r.get('backend', 'default')})" == lib
        ]
        if lib_results:
            avg_time = sum(r["mean_ms"] for r in lib_results) / len(lib_results)
            print(f"{lib:30s}: {avg_time:8.2f}ms average")


if __name__ == "__main__":
    main()
