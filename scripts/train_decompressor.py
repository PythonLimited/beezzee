"""Train the KV decompressor using a pre-trained chunker checkpoint.

Usage:
    python scripts/train_decompressor.py checkpoints/chunker_linear_k4_step5000.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chunkers import build_chunker
from src.kv_decompressor import KVDecompressor
from src.kv_trainer import KVDecompressorTrainer
from src.trainer import TextDataset


def main():
    MODEL_DIR = Path("models/Qwen_Qwen3.5-0.8B-Base")
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not ckpt_path or not ckpt_path.exists():
        print("Usage: python scripts/train_decompressor.py checkpoints/chunker_*.pt")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    K = ckpt["chunk_size"]
    print(f"Chunker checkpoint: step={ckpt['step']}, K={K}")

    print(f"Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float16,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
        device_map={"": device} if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    chunker = build_chunker(ckpt["chunker_type"], ckpt["hidden_dim"], K)
    chunker.load_state_dict(ckpt["chunker_state"])
    chunker = chunker.to(device=device, dtype=model.dtype)
    chunker.eval()

    D = model.config.hidden_size
    decompressor = KVDecompressor(D, K).to(device=device, dtype=model.dtype)
    decompressor.train()

    trainer = KVDecompressorTrainer(
        base_model=model,
        chunker=chunker,
        decompressor=decompressor,
        chunk_size=K,
    )

    STEPS = 2000
    seq_len = 256

    print(f"\n{'='*60}")
    print(f"  Decompressor: 3-layer MLP, K={K}")
    print(f"  Params: {sum(p.numel() for p in decompressor.parameters()):,}")
    print(f"  Steps: {STEPS}")
    print(f"{'='*60}\n")

    dataset = TextDataset(tokenizer, seq_len=seq_len)
    data_iter = iter(dataset)

    running_loss = 0.0
    for step in range(STEPS):
        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            dataset = TextDataset(tokenizer, seq_len=seq_len)
            data_iter = iter(dataset)
            input_ids = next(data_iter).to(device)

        metrics = trainer.train_step(input_ids)
        running_loss += metrics["loss"]

        if (step + 1) % 50 == 0:
            print(
                f"  Step {step+1:5d}/{STEPS} | "
                f"loss: {running_loss/50:.6f} | "
                f"lr: {metrics['lr']:.2e}"
            )
            running_loss = 0.0

        if (step + 1) % 200 == 0:
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            out = ckpt_dir / f"decompressor_k{K}_step{step+1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "decompressor_state": decompressor.state_dict(),
                    "chunk_size": K,
                    "hidden_dim": D,
                },
                out,
            )
            print(f"           saved → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
