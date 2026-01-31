"""Triton kernel utilities and constants."""

from __future__ import annotations

# Default block sizes for Triton kernels
DEFAULT_BLOCK_SIZE_N = 128
DEFAULT_BLOCK_SIZE_M = 128

# Numerical constants
LOG_EPSILON = 1e-38  # For log(0) protection
INF = float("inf")
