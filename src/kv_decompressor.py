"""
KV Decompressor: expands 1 compressed KV entry → K full entries.

Trained via self-distillation — MSE between decompressed and ground-truth
KV caches from a full prefill pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVDecompressor(nn.Module):
    """
    Small MLP that expands a compressed KV entry into K full-resolution
    entries, conditioned on the position offset within the chunk.

    Architecture: [D + pos_emb] → 4D → 2D → D  (3-layer MLP)
    """

    def __init__(self, dim: int, chunk_size: int = 4, hidden_mult: int = 4):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size

        # Learned position embeddings for offsets 0..K-1
        self.pos_emb = nn.Embedding(chunk_size, dim)

        # MLP: input = compressed_kv + pos_emb → expanded_kv
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, compressed_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            compressed_kv: [B, H, C, D] — C compressed KV entries
        Returns:
            expanded_kv: [B, H, C*K, D] — full KV entries
        """
        B, H, C, D = compressed_kv.shape
        K = self.chunk_size

        # Repeat each compressed entry K times, add position embedding
        # compressed_kv: [B, H, C, D] → [B, H, C, K, D] → [B, H, C*K, D]
        expanded = compressed_kv.unsqueeze(3).expand(B, H, C, K, D)

        # Build position offsets: 0, 1, ..., K-1 repeated C times
        offsets = torch.arange(K, device=compressed_kv.device).repeat(C)  # [C*K]
        pos = self.pos_emb(offsets)  # [C*K, D]
        pos = pos.view(1, 1, C * K, D).expand(B, H, -1, -1)  # [B, H, C*K, D]

        # Concatenate compressed_kv (repeated) with position embeddings
        flat = expanded.reshape(B, H, C * K, D)  # [B, H, C*K, D]
        inp = torch.cat([flat, pos], dim=-1)  # [B, H, C*K, 2D]

        # Apply MLP
        out = self.net(inp)  # [B, H, C*K, D]

        # Residual connection: add the naive repeat back
        return out + flat


class CompressorDecompressor(nn.Module):
    """
    Combined chunker + decompressor for end-to-end training.
    Chunker: N tokens → N/K compressed
    Decompressor: N/K compressed KV → N expanded KV
    """

    def __init__(self, dim: int, chunk_size: int = 4):
        super().__init__()
        self.chunk_size = chunk_size
        # Chunker is kept separate (trained via MSE on hidden states)
        # Decompressor is trained via MSE on KV caches
        self.decompressor = KVDecompressor(dim, chunk_size)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand compressed KV entries."""
        return self.decompressor(kv)
