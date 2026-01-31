"""Sinkhorn-Triton: High-performance Triton kernels for optimal transport."""

__version__ = "0.1.0"

from sinkhorn.sinkhorn import sinkhorn, SinkhornOutput

__all__ = ["sinkhorn", "SinkhornOutput", "__version__"]
