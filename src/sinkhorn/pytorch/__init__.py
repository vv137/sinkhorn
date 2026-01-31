"""Pure PyTorch implementations for reference and fallback."""

from sinkhorn.pytorch.balanced import sinkhorn_balanced
from sinkhorn.pytorch.unbalanced import sinkhorn_unbalanced

__all__ = ["sinkhorn_balanced", "sinkhorn_unbalanced"]
