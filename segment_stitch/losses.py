"""Segmentation losses."""

from __future__ import annotations


class DiceLoss(__import__("torch").nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        torch = __import__("torch")
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        return (1.0 - (2.0 * intersection + self.eps) / (union + self.eps)).mean()
