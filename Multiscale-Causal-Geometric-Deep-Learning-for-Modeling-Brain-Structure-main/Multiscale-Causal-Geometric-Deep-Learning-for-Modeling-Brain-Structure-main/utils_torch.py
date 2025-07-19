
"""Torch-based differentiable mutual information utilities."""
import torch

__all__ = ["MI", "calculate_sigma", "calculate_MI", "calculate_conditional_MI"]

_EPS = 1e-8
_ALPHA = 1.01

def _pairwise_sqdist(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, x, p=2).pow(2)

def calculate_sigma(x: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Kernel bandwidth (Silverman style): mean sqrt distance to k nearest neighbours."""
    with torch.no_grad():
        d = _pairwise_sqdist(x)
        return d.topk(k=k, largest=False).values.mean().sqrt().clamp_min(0.1)

def _gram(x: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    return torch.exp(-_pairwise_sqdist(x) / (sigma2 + _EPS))

def _renyi_entropy(kmat: torch.Tensor) -> torch.Tensor:
    kmat = kmat / (kmat.trace() + _EPS)
    eigvals = torch.linalg.eigvalsh(kmat).clamp_min(0.)
    return torch.log2(eigvals.pow(_ALPHA).sum() + _EPS) / (1.0 - _ALPHA)

def MI(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Kernel Rényi mutual information I(x;y)."""
    sx2 = calculate_sigma(x) ** 2
    sy2 = calculate_sigma(y) ** 2
    return (_renyi_entropy(_gram(x, sx2)) + 
            _renyi_entropy(_gram(y, sy2)) - 
            _renyi_entropy(_gram(x, sx2) * _gram(y, sy2)))

# Backward‑compatible aliases
calculate_MI = MI

def _to_2d(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() > 1 else t.unsqueeze(-1)

def calculate_conditional_MI(x: torch.Tensor,
                             z: torch.Tensor,
                             y: torch.Tensor) -> torch.Tensor:
    x2, z2, y2 = map(_to_2d, (x, z, y))
    return MI(torch.cat([x2, z2], dim=-1),
              torch.cat([y2, z2], dim=-1)) - MI(z2, z2)
