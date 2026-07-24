"""
MTC model wrapper: attaches a chunker before a HuggingFace model
for shorter prompt processing, then decompresses the KV cache for
standard autoregressive generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache, Cache

from src.chunkers import build_chunker


def compatible_position_ids(
    position_ids: torch.Tensor,
    chunk_size: int,
    long_context: bool = False,
) -> torch.Tensor:
    """
    Map N positions → ceil(N/K) position IDs.

    non-contiguous (default): [K-1, 2K-1, ...] — best quality at ≤32K
    contiguous (long_context): [0, 1, 2, ...] — flash-attn compatible for 65K+
    """
    N = position_ids.shape[-1]
    K = chunk_size

    if long_context:
        n_compressed = N // K + (1 if N % K else 0)
        return torch.arange(n_compressed, device=position_ids.device).unsqueeze(0)

    n_full = N // K
    pids = position_ids.squeeze(0)
    compressed = pids[:n_full * K].view(-1, K)[:, -1]
    remainder = N % K
    if remainder:
        compressed = torch.cat([compressed, pids[-remainder:][-1:]], dim=0)
    return compressed.unsqueeze(0)


class MTCModel(nn.Module):
    """
    Wraps a HuggingFace causal LM for multi-token prefill.

    During prefill:
      1. Embed prompt (N tokens)
      2. Chunk & compress → N/K compressed embeddings
      3. Map position IDs (last position of each chunk)
      4. Forward through base model → M = N/K hidden states + compressed KV cache
      5. Decompress KV cache → N entries (by repeating)
      6. Use expanded KV + last hidden state for generation

    During generation: standard autoregressive (uses expanded cache).
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        chunk_name: str = "mean",
        chunk_size: int = 4,
        decompressor: nn.Module = None,
    ):
        super().__init__()
        self.base = base_model
        self.tokenizer = tokenizer
        self._chunk_size = chunk_size  # bypass setter until chunker exists
        self.hidden_dim = base_model.config.hidden_size
        self.chunker = build_chunker(chunk_name, self.hidden_dim, chunk_size)
        self.decompressor = decompressor

        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

    @property
    def chunk_size(self):
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int):
        self._chunk_size = value
        self.chunker.chunk_size = value

    @property
    def config(self):
        return self.base.config

    @property
    def device(self):
        return next(self.base.parameters()).device

    def prefill(self, input_ids: torch.Tensor, long_context: bool = False) -> tuple[torch.Tensor, DynamicCache]:
        """Compressed prefill.

        Args:
            long_context: if True, use contiguous positions + flash-attn
                          for 65K+ sequences (slightly lower quality).
        """
        B, N = input_ids.shape
        K = self.chunk_size

        with torch.no_grad():
            embeddings = self.base.get_input_embeddings()(input_ids)
            compressed = self.chunker(embeddings)
            model_dtype = next(self.base.parameters()).dtype

            pos_ids = torch.arange(N, device=input_ids.device).unsqueeze(0)
            pos_ids_compressed = compatible_position_ids(pos_ids, K, long_context)

            outputs = self.base.model(
                inputs_embeds=compressed.to(model_dtype),
                position_ids=pos_ids_compressed,
                use_cache=True,
                past_key_values=None,
            )

        cache: DynamicCache = outputs.past_key_values

        # Expand full-attention KV via trained decompressor (if available)
        if self.decompressor is not None:
            expanded = DynamicCache()
            for layer_idx, layer in enumerate(cache.layers):
                if hasattr(layer, "keys") and hasattr(layer, "values"):
                    k_exp = self.decompressor(layer.keys)[:, :, :N, :]
                    v_exp = self.decompressor(layer.values)[:, :, :N, :]
                    expanded.update(k_exp, v_exp, layer_idx)
                else:
                    while len(expanded.layers) <= layer_idx:
                        expanded.layers.append(None)
                    expanded.layers[layer_idx] = layer
            cache = expanded

        last_hidden = outputs.last_hidden_state[:, -1:, :]
        logits = self.base.lm_head(last_hidden)
        return logits, cache

    def standard_prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, DynamicCache]:
        """Standard (uncompressed) prefill for comparison."""
        with torch.no_grad():
            outputs = self.base(input_ids=input_ids, use_cache=True)
        return outputs.logits[:, -1:, :], outputs.past_key_values

    def generate_from_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Standard autoregressive generation from an existing KV cache."""
        generated = []
        next_input = input_ids[:, -1:]

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.base(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            generated.append(next_token.item())
            next_input = next_token
            past_key_values = outputs.past_key_values

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return torch.tensor([generated], device=input_ids.device)
