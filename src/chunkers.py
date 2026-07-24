from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


def _interp_decompress(kv: torch.Tensor, chunk_size: int, N: int = None) -> torch.Tensor:
    """
    Expand compressed KV by interpolating between adjacent anchors.

    Instead of naive repeating (causes attention collapse), each decompressed
    position gets a unique KV value via linear interpolation between its two
    nearest compressed anchors. This preserves smooth attention patterns.
    """
    B, H, C, D = kv.shape
    K = chunk_size
    N = N or C * K

    expanded = torch.zeros(B, H, N, D, device=kv.device, dtype=kv.dtype)

    for i in range(N):
        c = i // K
        offset = i % K

        if c == 0:
            if C > 1:
                t = offset / K
                expanded[:, :, i, :] = (1 - t) * kv[:, :, 0, :] + t * kv[:, :, 1, :]
            else:
                expanded[:, :, i, :] = kv[:, :, 0, :]
        elif c >= C:
            expanded[:, :, i, :] = kv[:, :, -1, :]
        else:
            t = (offset + 1) / (K + 1)
            expanded[:, :, i, :] = (1 - t) * kv[:, :, c - 1, :] + t * kv[:, :, c, :]

    return expanded


class MeanChunk(nn.Module):
    def __init__(self, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        K = self.chunk_size
        n_full = N // K
        x = x[:, :n_full * K, :]
        return x.view(B, -1, K, D).mean(dim=2)

    def decompress_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Interpolate between compressed anchors for smooth decompression."""
        return _interp_decompress(kv, self.chunk_size)


class AttnChunk(nn.Module):
    def __init__(self, dim: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.query = nn.Parameter(torch.randn(1, 1, 1, dim) * 0.02)
        self.scale = dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        K = self.chunk_size
        n_full = N // K
        x = x[:, :n_full * K, :]
        chunks = x.view(B, -1, K, D)
        attn = (self.query * chunks).sum(dim=-1) * self.scale  # [B, n_chunks, K]
        attn = F.softmax(attn, dim=-1)
        return (chunks * attn.unsqueeze(-1)).sum(dim=2)

    def decompress_kv(self, kv: torch.Tensor) -> torch.Tensor:
        return kv.repeat_interleave(self.chunk_size, dim=2)


class LinearChunk(nn.Module):
    def __init__(self, dim: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.proj = nn.Linear(dim * chunk_size, dim, bias=False)
        self._init_as_mean_pool()

    def _init_as_mean_pool(self):
        D = self.proj.out_features
        K = self.chunk_size
        weight = torch.zeros(D * K, D)
        for k in range(K):
            weight[k * D : (k + 1) * D, :] = torch.eye(D) / K
        self.proj.weight.data = weight.T.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        K = self.chunk_size
        n_full = N // K
        x = x[:, :n_full * K, :]
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
