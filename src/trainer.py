"""
Self-distillation training: joint training of chunker + decompressor.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from src.mtc_model import compatible_position_ids


def _is_main():
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


class MTCTrainer:
    """Joint training of chunker (semantic) + decompressor (generation)."""

    def __init__(
        self,
        base_model: nn.Module,
        chunker: nn.Module,
        chunk_size: int = 4,
        decompressor: nn.Module = None,
    ):
        self.base = base_model
        self.chunker = chunker
        self.decompressor = decompressor
        self.chunk_size = chunk_size

        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

        self._compiled_model = torch.compile(self.base.model, mode="default", fullgraph=False)

        params = list(self.chunker.parameters())
        if decompressor is not None:
            params += list(decompressor.parameters())
        label = "chunker+decompressor" if decompressor else "chunker"
        print(f"Trainable params: {sum(p.numel() for p in params):,} ({label})")

        self.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15000, eta_min=1e-6
        )

    @torch.no_grad()
    def _model(self, embeds, pos_ids, use_cache=False):
        out = self._compiled_model(inputs_embeds=embeds, position_ids=pos_ids, use_cache=use_cache)
        return out.last_hidden_state, out.past_key_values

    @torch.no_grad()
    def _teacher_chunked(self, input_ids):
        B, N = input_ids.shape
        K = self.chunk_size
        embeds = self.base.get_input_embeddings()(input_ids)
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)
        full, kv = self._model(embeds, pos, use_cache=self.decompressor is not None)

        n_full = N // K
        chunked = full[:, :n_full * K, :].view(B, n_full, K, -1)[:, :, -1, :]
        if N % K:
            chunked = torch.cat([chunked, full[:, -1:, :]], dim=1)
        return (chunked, kv) if self.decompressor is not None else chunked

    @torch.no_grad()
    def _student_forward(self, compressed, N):
        pos = torch.arange(N, device=compressed.device).unsqueeze(0)
        pos_c = compatible_position_ids(pos, self.chunk_size)
        dtype = next(self.base.parameters()).dtype
        return self._model(compressed.to(dtype), pos_c, use_cache=self.decompressor is not None)

    def train_step(self, input_ids):
        B, N = input_ids.shape
        K = self.chunk_size

        # Teacher
        if self.decompressor is not None:
            teacher_hidden, teacher_kv = self._teacher_chunked(input_ids)
        else:
            teacher_hidden = self._teacher_chunked(input_ids)

        # Student
        embeddings = self.base.get_input_embeddings()(input_ids).detach()
        compressed = self.chunker(embeddings)
        student_hidden, student_kv = self._student_forward(compressed, N)

        # Chunker loss: MSE on hidden states (straight-through proxy)
        loss_hidden = F.mse_loss(student_hidden, teacher_hidden)
        grad_output = (student_hidden - teacher_hidden) * (2.0 / student_hidden.numel())
        proxy_loss = (compressed * grad_output.to(compressed.dtype).detach()).sum()

        # Decompressor loss: MSE on KV caches
        loss_kv = torch.tensor(0.0, device=input_ids.device)
        if self.decompressor is not None:
            n_layers = 0
            for comp_layer, gt_layer in zip(student_kv.layers, teacher_kv.layers):
                if not (hasattr(comp_layer, "keys") and hasattr(gt_layer, "keys")):
                    continue
                exp_k = self.decompressor(comp_layer.keys)[:, :, :N, :]
                exp_v = self.decompressor(comp_layer.values)[:, :, :N, :]
                loss_kv += F.mse_loss(exp_k, gt_layer.keys) + F.mse_loss(exp_v, gt_layer.values)
                n_layers += 1
            if n_layers > 0:
                loss_kv = loss_kv / n_layers

        # Top-1 monitoring
        with torch.no_grad():
            t_logits = self.base.lm_head(teacher_hidden[:, -1:, :])
            s_logits = self.base.lm_head(student_hidden[:, -1:, :])
            top1 = (t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean().item()

        return {
            "loss": loss_hidden.item() / student_hidden.shape[-1],
            "loss_kv": loss_kv.item(),
            "top1_match": top1,
            "lr": self.scheduler.get_last_lr()[0],
            "_proxy_loss": proxy_loss,
        }


# ── Data ────────────────────────────────────────────────────────────


class TextDataset(IterableDataset):
    """Lazy-batched tokenization, yields fixed-length tensor slices."""

    SOURCES = [("Salesforce/wikitext", "wikitext-103-raw-v1", "train", 1.0, "text", None)]
    CACHE_DIR = Path("datasets")

    def __init__(self, tokenizer, seq_len=256, rank=0, world_size=1):
        from datasets import load_dataset, load_from_disk

        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.CACHE_DIR.mkdir(exist_ok=True)

        data_sources = []
        for path, name, split, weight, field, fraction in self.SOURCES:
            cache_path = self.CACHE_DIR / path.replace("/", "_")
            if name:
                cache_path = cache_path.with_name(f"{cache_path.name}_{name}")
            cache_path = cache_path / (f"{split}_{fraction.replace('%','pct')}" if fraction else split)

            try:
                if cache_path.exists():
                    ds = load_from_disk(str(cache_path))
                else:
                    if _is_main():
                        print(f"  Downloading {path}/{name or ''} → {cache_path} ...")
                    ds_split = f"{split}[{fraction}]" if fraction else split
                    ds = load_dataset(path, name, split=ds_split) if name else load_dataset(path, split=ds_split)
                    ds.save_to_disk(str(cache_path))
                data_sources.append((ds, field))
                if _is_main():
                    print(f"  ✓ {path}/{name or ''}  ({cache_path})")
            except Exception as e:
                if _is_main():
                    print(f"  ✗ {path}/{name or ''}: {e}")

        self._sources = data_sources
        self._tokenizer = tokenizer

    def __iter__(self):
        sl, rank, world, chunk_size = self.seq_len, self.rank, self.world_size, 5000
        buffer, slice_idx = [], rank

        for ds, field in self._sources:
            batch = []
            for sample in ds:
                text = sample.get(field, "")
                if text and text.strip():
                    batch.append(text)
                if len(batch) >= chunk_size:
                    for ids in self._tokenizer(batch, add_special_tokens=False).input_ids:
                        buffer.extend(ids)
                    batch = []
                    while slice_idx * sl + sl <= len(buffer):
                        start = slice_idx * sl
                        yield torch.tensor(buffer[start:start + sl], dtype=torch.long).unsqueeze(0)
                        slice_idx += world
            if batch:
                for ids in self._tokenizer(batch, add_special_tokens=False).input_ids:
                    buffer.extend(ids)

        total = len(buffer)
        while True:
            while slice_idx * sl + sl <= total:
                start = slice_idx * sl
                yield torch.tensor(buffer[start:start + sl], dtype=torch.long).unsqueeze(0)
                slice_idx += world
            slice_idx = rank


# ── Eval ────────────────────────────────────────────────────────────


@torch.no_grad()
def eval_step(trainer, input_ids):
    teacher = trainer._teacher_chunked(input_ids)
    if trainer.decompressor is not None:
        teacher = teacher[0]

    embeddings = trainer.base.get_input_embeddings()(input_ids).detach()
    compressed = trainer.chunker(embeddings)
    student = trainer._student_forward(compressed, input_ids.shape[1])[0]

    loss = F.mse_loss(student, teacher)
    t_logits = trainer.base.lm_head(teacher[:, -1:, :])
    s_logits = trainer.base.lm_head(student[:, -1:, :])
    top1 = (t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean().item()
    return {"eval_loss": loss.item() / student.shape[-1], "eval_top1": top1}
