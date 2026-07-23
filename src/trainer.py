"""
Self-distillation training: teach the chunker to compress prompts
while preserving the base model's hidden-state distribution.

Uses a straight-through estimator — gradient flows only through the
chunker, avoiding fp16 numerical instability in the frozen model.
Requires no labeled data — any raw text works.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from src.mtc_model import compatible_position_ids


class MTCTrainer:
    """
    Trains only the chunker via a straight-through estimator.

    Gradient flow through the frozen fp16 model is numerically unstable.
    Instead we:
      1. Compute teacher & student hidden states under no_grad
      2. Use the loss gradient w.r.t. hidden states as a proxy gradient
         for the chunker output (approximating ∂model/∂input ≈ I)
      3. Only backprop through the chunker (fp32, stable)
    """

    def __init__(
        self,
        base_model: nn.Module,
        chunker: nn.Module,
        chunk_size: int = 4,
    ):
        self.base = base_model
        self.chunker = chunker
        self.chunk_size = chunk_size

        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

        # Compile the model for faster no_grad forward passes
        self._compiled_model = torch.compile(
            self.base.model, mode="reduce-overhead", fullgraph=False
        )

        params = list(self.chunker.parameters())
        print(f"Trainable chunker params: {sum(p.numel() for p in params):,}")

        self.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=1e-6
        )

    @torch.no_grad()
    def _model_forward(self, embeds: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        out = self._compiled_model(inputs_embeds=embeds, position_ids=pos_ids, use_cache=False)
        return out.last_hidden_state

    @torch.no_grad()
    def _teacher_chunked(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full forward → chunked hidden states matching student positions."""
        B, N = input_ids.shape
        K = self.chunk_size
        embeds = self.base.get_input_embeddings()(input_ids)
        pos_ids = torch.arange(N, device=input_ids.device).unsqueeze(0)
        full = self._model_forward(embeds, pos_ids)

        n_full = N // K
        chunked = full[:, :n_full * K, :].view(B, n_full, K, -1)[:, :, -1, :]
        remainder = N % K
        if remainder:
            chunked = torch.cat([chunked, full[:, -1:, :]], dim=1)
        return chunked

    @torch.no_grad()
    def _student_forward(self, compressed: torch.Tensor, N: int) -> torch.Tensor:
        """Model forward on compressed embeddings → hidden states."""
        pos_ids = torch.arange(N, device=compressed.device).unsqueeze(0)
        pos_comp = compatible_position_ids(pos_ids, self.chunk_size)
        return self._model_forward(compressed, pos_comp)

    def train_step(self, input_ids: torch.Tensor) -> dict:
        B, N = input_ids.shape
        K = self.chunk_size
        dtype = next(self.base.parameters()).dtype

        # 1. Teacher target (no grad)
        teacher_hidden = self._teacher_chunked(input_ids)

        # 2. Student: chunker forward (WITH grad), then model forward (no grad)
        embeddings = self.base.get_input_embeddings()(input_ids).detach()
        compressed = self.chunker(embeddings)
        student_hidden = self._student_forward(compressed, N)

        # 3. MSE loss
        loss = F.mse_loss(student_hidden, teacher_hidden)

        # 4. Straight-through: use loss gradient at hidden states as
        #    proxy gradient for chunker output (≈ identity Jacobian through model)
        grad_output = (student_hidden - teacher_hidden) * (2.0 / student_hidden.numel())

        # 5. Backprop through chunker only
        self.optimizer.zero_grad()
        compressed.backward(gradient=grad_output)
        nn.utils.clip_grad_norm_(self.chunker.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        with torch.no_grad():
            t_logits = self.base.lm_head(teacher_hidden[:, -1:, :])
            s_logits = self.base.lm_head(student_hidden[:, -1:, :])
            top1 = (t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean().item()

        return {"loss": loss.item(), "top1_match": top1, "lr": self.scheduler.get_last_lr()[0]}


# ── Data ────────────────────────────────────────────────────────────


class TextDataset(IterableDataset):
    """
    Lazy-batched tokenization: loads text in chunks, batch-tokenizes via
    the Rust tokenizer, yields fixed-length tensor slices. No up-front cost.
    """

    SOURCES = [
        ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", 1.0, "text", None),
    ]

    CACHE_DIR = Path("datasets")

    def __init__(self, tokenizer, seq_len: int = 256):
        from datasets import load_dataset, load_from_disk

        self.seq_len = seq_len
        self.CACHE_DIR.mkdir(exist_ok=True)

        data_sources = []
        for path, name, split, weight, text_field, fraction in self.SOURCES:
            cache_path = self.CACHE_DIR / path.replace("/", "_")
            if name:
                cache_path = cache_path.with_name(f"{cache_path.name}_{name}")
            if fraction:
                cache_path = cache_path / f"{split}_{fraction.replace('%','pct')}"
            else:
                cache_path = cache_path / split

            try:
                if cache_path.exists():
                    ds = load_from_disk(str(cache_path))
                else:
                    label = f"{path}/{name or ''}" + (f" ({fraction})" if fraction else "")
                    print(f"  Downloading {label} → {cache_path} ...")
                    ds_split = f"{split}[{fraction}]" if fraction else split
                    if name:
                        ds = load_dataset(path, name, split=ds_split)
                    else:
                        ds = load_dataset(path, split=ds_split)
                    ds.save_to_disk(str(cache_path))
                data_sources.append((ds, text_field))
                print(f"  ✓ {path}/{name or ''}  ({cache_path})")
            except Exception as e:
                print(f"  ✗ {path}/{name or ''}: {e}")

        self._sources = data_sources
        self._tokenizer = tokenizer

    def __iter__(self):
        buffer = []
        sl = self.seq_len
        chunk_size = 5000  # batch-tokenize this many texts at once

        for ds, field in self._sources:
            texts_batch = []
            for sample in ds:
                text = sample.get(field, "")
                if text and text.strip():
                    texts_batch.append(text)
                if len(texts_batch) >= chunk_size:
                    for ids in self._tokenizer(texts_batch, add_special_tokens=False).input_ids:
                        buffer.extend(ids)
                        while len(buffer) >= sl:
                            yield torch.tensor(buffer[:sl], dtype=torch.long).unsqueeze(0)
                            buffer = buffer[sl:]
                    texts_batch = []
            # Flush remaining
            if texts_batch:
                for ids in self._tokenizer(texts_batch, add_special_tokens=False).input_ids:
                    buffer.extend(ids)
                    while len(buffer) >= sl:
                        yield torch.tensor(buffer[:sl], dtype=torch.long).unsqueeze(0)
                        buffer = buffer[sl:]

        # Wrap around forever from buffer residue
        wrapped = list(buffer)
        while True:
            buffer = wrapped[:]
            wrapped = []
            while len(buffer) >= sl:
                yield torch.tensor(buffer[:sl], dtype=torch.long).unsqueeze(0)
                buffer = buffer[sl:]
            # Refill
            wrapped = list(buffer)


# ── Eval ────────────────────────────────────────────────────────────


@torch.no_grad()
def eval_step(trainer: MTCTrainer, input_ids: torch.Tensor) -> dict:
    teacher = trainer._teacher_chunked(input_ids)
    embeddings = trainer.base.get_input_embeddings()(input_ids).detach()
    compressed = trainer.chunker(embeddings)
    student = trainer._student_forward(compressed, input_ids.shape[1])

    loss = F.mse_loss(student, teacher)
    t_logits = trainer.base.lm_head(teacher[:, -1:, :])
    s_logits = trainer.base.lm_head(student[:, -1:, :])
    top1 = (t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean().item()
    return {"eval_loss": loss.item(), "eval_top1": top1}
