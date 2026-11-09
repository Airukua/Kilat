import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input tensor along the last dimension by dividing by the
    root mean square of its elements, then applies a learnable per‑dimension
    scale. Unlike standard LayerNorm, it omits the mean subtraction step,
    making it more memory‑efficient and slightly faster, while maintaining
    comparable training stability.
    """

    def __init__(self, normalized_shape, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_up = x.to(compute_dtype)
        variance = x_up.pow(2).mean(dim=-1, keepdim=True)
        x_normed = (x_up * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return x_normed * self.weight
