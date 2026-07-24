"""Train chunker via autoencoder: compress K embeddings → 1 → reconstruct K.

No LLM forward passes — trains on pure embedding reconstruction.
Learns to weight important tokens higher than filler words.

Usage:
    python scripts/train_chunker_ae.py --chunk-size 4 --steps 5000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class ChunkAutoencoder(nn.Module):
    """K embeddings → 1 compressed → K reconstructions."""

    def __init__(self, dim: int, chunk_size: int, bottleneck: int | None = None):
        super().__init__()
        self.chunk_size = chunk_size
        bottleneck = bottleneck or dim
        self.compress = nn.Linear(dim * chunk_size, bottleneck, bias=False)
        self.expand = nn.Linear(bottleneck, dim * chunk_size, bias=False)
        self._init_as_mean_pool()

    def _init_as_mean_pool(self):
        D = self.compress.out_features
        K = self.chunk_size
        if self.compress.out_features == self.expand.in_features:
            weight = torch.zeros(D * K, D)
            for k in range(K):
                weight[k * D : (k + 1) * D, :] = torch.eye(D) / K
            self.compress.weight.data = weight.T.contiguous()
            self.expand.weight.data = weight.contiguous()

    def encode(self, chunks: torch.Tensor) -> torch.Tensor:
        """[B, N/K, K*D] → [B, N/K, bottleneck]"""
        return self.compress(chunks)

    def decode(self, compressed: torch.Tensor) -> torch.Tensor:
        """[B, N/K, bottleneck] → [B, N/K, K*D]"""
        return self.expand(compressed)

    def forward(self, chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (compressed, reconstructed)."""
        c = self.compress(chunks)
        r = self.expand(c)
        return c, r


class EmbeddingDataset(IterableDataset):
    """Tokenize WikiText → lookup embeddings → yield chunks."""

    def __init__(self, tokenizer, model, seq_len: int = 512, rank: int = 0,
                 world_size: int = 1):
        from datasets import load_dataset, load_from_disk

        cache_dir = Path("datasets/Salesforce_wikitext_wikitext-103-raw-v1/train")
        cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_dir.exists() and any(cache_dir.iterdir()):
            ds = load_from_disk(str(cache_dir))
        else:
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                              split="train")
            ds.save_to_disk(str(cache_dir))

        self.ds = ds
        self.tokenizer = tokenizer
        self.model = model
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        sl = self.seq_len
        buffer = []
        slice_idx = self.rank

        for sample in self.ds:
            text = sample.get("text", "")
            if text and text.strip():
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                buffer.extend(ids)
                while slice_idx * sl + sl <= len(buffer):
                    start = slice_idx * sl
                    seq = torch.tensor(buffer[start:start + sl], dtype=torch.long)
                    slice_idx += self.world_size

                    with torch.no_grad():
                        embeds = self.model.get_input_embeddings()(seq.unsqueeze(0))
                    yield embeds.squeeze(0)  # [N, D]

        while True:
            while slice_idx * sl + sl <= len(buffer):
                start = slice_idx * sl
                seq = torch.tensor(buffer[start:start + sl], dtype=torch.long)
                slice_idx += self.world_size
                with torch.no_grad():
                    embeds = self.model.get_input_embeddings()(seq.unsqueeze(0))
                yield embeds.squeeze(0)
            slice_idx = self.rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bottleneck", type=int, default=None,
                        help="Compression dim (default: same as model dim)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    MODEL_DIR = Path("models/Qwen_Qwen3.5-0.8B-Base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = args.chunk_size

    print(f"Device: {device}  K={K}  steps={args.steps}  lr={args.lr}")
    print(f"Loading model from {MODEL_DIR} ...")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    dim = model.config.hidden_size
    bottleneck = args.bottleneck or dim
    ae = ChunkAutoencoder(dim, K, bottleneck).to(device=device, dtype=torch.float32)
    ae.train()

    optimizer = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps * 2, eta_min=1e-5
    )

    print(f"  AE params:  compress {ae.compress.weight.shape}  "
          f"expand {ae.expand.weight.shape}")
    print(f"  Total: {sum(p.numel() for p in ae.parameters()):,}")
    print(f"  Bottleneck: {bottleneck} ({bottleneck/dim*100:.0f}% of {dim})")
    print()

    dataset = EmbeddingDataset(tokenizer, model, seq_len=args.seq_len)
    data_iter = iter(dataset)

    running_loss = 0.0
    best_loss = float("inf")

    for step in range(args.steps):
        try:
            embeds = next(data_iter).to(device)  # [N, D]
        except StopIteration:
            dataset = EmbeddingDataset(tokenizer, model, seq_len=args.seq_len)
            data_iter = iter(dataset)
            embeds = next(data_iter).to(device)

        N = (embeds.shape[0] // K) * K
        embeds = embeds[:N]
        chunks = embeds.view(N // K, K, -1).reshape(N // K, K * dim)  # [N/K, K*D]

        compressed, reconstructed = ae(chunks)

        loss = F.mse_loss(reconstructed, chunks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if (step + 1) % 100 == 0:
            avg = running_loss / 100
            pct = 100 * (step + 1) / args.steps
            rel_improve = (1 - avg / max(running_loss / 100, 0.0001)) * 100 if step == 0 else 0
            print(f"  step {step+1:5d}/{args.steps} ({pct:3.0f}%)  "
                  f"recon_loss: {avg:.6f}  lr: {scheduler.get_last_lr()[0]:.2e}")

            if avg < best_loss:
                best_loss = avg

            running_loss = 0.0

        if (step + 1) % 500 == 0:
            out_path = args.output or f"checkpoints/chunker_ae_k{K}_step{step+1}.pt"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step + 1,
                "chunker_state": {"proj.weight": ae.compress.weight.data.clone()},
                "chunker_type": "linear",
                "chunk_size": K,
                "hidden_dim": dim,
                "kv_head_dim": dim,  # placeholder
                "ae_recon_loss": best_loss,
            }, out_path)
            print(f"  saved -> {out_path}")

    print(f"\nDone. Best recon loss: {best_loss:.6f}")
    print(f"Load with: --chunker checkpoints/chunker_ae_k{K}_step{args.steps}.pt")


if __name__ == "__main__":
    main()
