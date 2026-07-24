"""
Train the KV decompressor: learn to expand compressed KV entries
back to full resolution for generation-quality parity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVDecompressorTrainer:
    """
    Trains the decompressor via MSE between expanded and ground-truth KV.

    The chunker and base model are frozen. Only decompressor params train.

    For each text sequence:
      1. Standard prefill → ground-truth KV (full-attention layers)
      2. Compressed prefill → compressed KV → decompress → MSE loss
    """

    def __init__(
        self,
        base_model: nn.Module,
        chunker: nn.Module,
        decompressor: nn.Module,
        chunk_size: int = 4,
    ):
        from src.mtc_model import compatible_position_ids

        self.base = base_model
        self.chunker = chunker
        self.decompressor = decompressor
        self.chunk_size = chunk_size
        self._compatible_pos = compatible_position_ids

        self.base.eval()
        self.chunker.eval()
        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.chunker.parameters():
            p.requires_grad = False

        params = list(self.decompressor.parameters())
        print(f"Trainable decompressor params: {sum(p.numel() for p in params):,}")
        self.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=1e-6
        )

    @torch.no_grad()
    def _model_forward(self, embeds: torch.Tensor, pos_ids: torch.Tensor) -> tuple:
        out = self.base.model(inputs_embeds=embeds, position_ids=pos_ids, use_cache=True)
        return out.last_hidden_state, out.past_key_values

    def train_step(self, input_ids: torch.Tensor) -> dict:
        B, N = input_ids.shape
        K = self.chunk_size
        dtype = next(self.base.parameters()).dtype
        device = input_ids.device

        # 1. Ground truth: full prefill → KV cache
        with torch.no_grad():
            embeds_full = self.base.get_input_embeddings()(input_ids)
            pos_full = torch.arange(N, device=device).unsqueeze(0)
            _, gt_cache = self._model_forward(embeds_full, pos_full)

        # 2. Compressed prefill → compressed KV
        with torch.no_grad():
            compressed = self.chunker(embeds_full.to(self.chunker.proj.weight.dtype))
            pos_comp = self._compatible_pos(torch.arange(N, device=device).unsqueeze(0), K)
            _, comp_cache = self._model_forward(compressed.to(dtype), pos_comp)

        # 3. For each full-attention layer, decompress and compute MSE
        total_loss = 0.0
        n_layers = 0
        for layer_idx, (comp_layer, gt_layer) in enumerate(
            zip(comp_cache.layers, gt_cache.layers)
        ):
            if not (hasattr(comp_layer, "keys") and hasattr(gt_layer, "keys")):
                continue
            if not (hasattr(comp_layer, "values") and hasattr(gt_layer, "values")):
                continue

            comp_k = comp_layer.keys   # [B, H, C, D]
            comp_v = comp_layer.values
            gt_k = gt_layer.keys       # [B, H, N, D]
            gt_v = gt_layer.values

            # Decompress
            exp_k = self.decompressor(comp_k)[:, :, :N, :]
            exp_v = self.decompressor(comp_v)[:, :, :N, :]

            total_loss += F.mse_loss(exp_k, gt_k) + F.mse_loss(exp_v, gt_v)
            n_layers += 1

        loss = total_loss / max(n_layers, 1)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.decompressor.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
