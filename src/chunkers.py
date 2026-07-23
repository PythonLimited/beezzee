"""
Multi-Token Consumption (MTC) — prompt processing speedup.

Strategy: compress N prompt token embeddings → N/K embeddings via learned
chunking, run transformer at reduced length, decompress KV cache for
standard autoregressive generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class MeanChunk(nn.Module):
    """Mean-pool every K embeddings. No params — baseline."""

    def __init__(self, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        K = self.chunk_size
        if N % K != 0:
            pad = K - (N % K)
            x = F.pad(x, (0, 0, 0, pad), mode="replicate")
        return x.view(B, -1, K, D).mean(dim=2)

    def decompress_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand compressed KV entries by repeating each K times."""
        B, H, compressed_len, D = kv.shape
        return kv.repeat_interleave(self.chunk_size, dim=2)


class AttnChunk(nn.Module):
    """Learnable attention-pooling: learns a query that attends over K tokens."""

    def __init__(self, dim: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.query = nn.Parameter(torch.randn(1, 1, 1, dim) * 0.02)
        self.scale = dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        K = self.chunk_size
        if N % K != 0:
            pad = K - (N % K)
            x = F.pad(x, (0, 0, 0, pad), mode="replicate")
        chunks = x.view(B, -1, K, D)
        attn = (self.query * chunks).sum(dim=-1) * self.scale  # [B, n_chunks, K]
        attn = F.softmax(attn, dim=-1)
        return (chunks * attn.unsqueeze(-1)).sum(dim=2)

    def decompress_kv(self, kv: torch.Tensor) -> torch.Tensor:
        return kv.repeat_interleave(self.chunk_size, dim=2)


class LinearChunk(nn.Module):
    """Learnable linear projection: concatenate K tokens → project to 1.

    Initialized to approximate mean pooling so training starts from a
    working baseline rather than random noise."""

    def __init__(self, dim: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.proj = nn.Linear(dim * chunk_size, dim, bias=False)
        self._init_as_mean_pool()

    def _init_as_mean_pool(self):
        """Set weights so output ≈ mean of K input slices."""
        D = self.proj.out_features
        K = self.chunk_size
        weight = torch.zeros(D * K, D)
        for k in range(K):
            weight[k * D : (k + 1) * D, :] = torch.eye(D) / K
        self.proj.weight.data = weight.T.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        K = self.chunk_size
        if N % K != 0:
            pad = K - (N % K)
            x = F.pad(x, (0, 0, 0, pad), mode="replicate")
        chunks = x.view(B, -1, K * D)
        return self.proj(chunks)

    def decompress_kv(self, kv: torch.Tensor) -> torch.Tensor:
        return kv.repeat_interleave(self.chunk_size, dim=2)


CHUNKERS = {"mean": MeanChunk, "attention": AttnChunk, "linear": LinearChunk}


def build_chunker(name: str, dim: int, chunk_size: int) -> nn.Module:
    cls = CHUNKERS[name]
    if name == "mean":
        return cls(chunk_size)
    return cls(dim, chunk_size)
