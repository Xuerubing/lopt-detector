"""Optional PyTorch reference feature extraction."""

from __future__ import annotations

import torch


def global_logm2(tensor: torch.Tensor, eps: float = 1e-30) -> float:
    value = torch.mean(tensor.detach().float().pow(2))
    return float(torch.log10(value + eps).item())
