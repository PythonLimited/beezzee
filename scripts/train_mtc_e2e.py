"""End-to-end training of chunker + model LoRA + decompressor.

Real gradients — no straight-through noise. The LoRA layers adapt the
model to understand compressed inputs, like teaching it a new "compressed
language" where each token represents K original tokens.

Usage:
    python scripts/train_mtc_e2e.py --steps 10000
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from src.chunkers import build_chunker
from src.kv_decompressor import KVDecompressor
from src.trainer import TextDataset


def _full_attention_layers(model):
    try:
        layer_types = model.config.layer_types
        return {i for i, t in enumerate(layer_types) if t == "full_attention"}
    except AttributeError:
        try:
            layer_types = model.config.text_config.layer_types
            return {i for i, t in enumerate(layer_types) if t == "full_attention"}
        except AttributeError:
            return set(range(model.config.num_hidden_layers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    K = args.chunk_size
    MODEL_DIR = Path("models/Qwen_Qwen3.5-0.8B-Base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    print(f"Device: {device}  K={K}  lora_rank={args.lora_rank}")
    print(f"Loading model from {MODEL_DIR} ...")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
        low_cpu_mem_usage=True,
    ).to(device)

    full_attn = _full_attention_layers(model)

    # Apply LoRA to full-attention layers' q_proj, v_proj
    target_modules = []
    for idx in sorted(full_attn):
        target_modules.extend([
            f"model.layers.{idx}.self_attn.q_proj",
            f"model.layers.{idx}.self_attn.v_proj",
        ])

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA params: {trainable:,}  (full-attn layers: {sorted(full_attn)})")

    # Chunker + decompressor
    dim = model.get_base_model().config.hidden_size
    head_dim = model.get_base_model().config.head_dim

    chunker = build_chunker("linear", dim, K).to(device=device, dtype=torch.float32)
    decompressor = KVDecompressor(head_dim, K, depth=5).to(device=device, dtype=dtype)

    chunker.train()
    decompressor.train()

    all_params = (list(chunker.parameters()) + list(decompressor.parameters()) +
                  [p for p in model.parameters() if p.requires_grad])
    print(f"  Chunker: {sum(p.numel() for p in chunker.parameters()):,}")
    print(f"  Decompressor: {sum(p.numel() for p in decompressor.parameters()):,}")
    print(f"  Total trainable: {sum(p.numel() for p in all_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": chunker.parameters(), "lr": args.lr * 10},
        {"params": decompressor.parameters(), "lr": args.lr * 10},
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-6
    )

    # Teacher model (frozen, full-context)
    teacher = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    dataset = TextDataset(tokenizer, seq_len=args.seq_len)
    data_iter = iter(dataset)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        chunker.load_state_dict(ckpt["chunker_state"])
        decompressor.load_state_dict(ckpt["decompressor_state"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    running_loss = 0.0
    running_kl = 0.0

    for step in range(start_step, args.steps):
        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            dataset = TextDataset(tokenizer, seq_len=args.seq_len)
            data_iter = iter(dataset)
            input_ids = next(data_iter).to(device)

        B, N_full = input_ids.shape
        N = (N_full // K) * K
        input_ids = input_ids[:, :N]
        base = model.get_base_model()

        # 1. Teacher: full forward → next-token logits
        with torch.no_grad():
            t_out = teacher(input_ids=input_ids)
            # Teacher logits: predict next token after seeing all N tokens
            t_logits = t_out.logits[:, -1, :]  # [B, V]

        # 2. Student: compress → model+LoRA → next-token logits
        embeds = base.get_input_embeddings()(input_ids)
        compressed = chunker(embeds)  # [B, N/K, D]

        # Sparse position IDs so chunks align with teacher
        pos = torch.arange(N, device=device).unsqueeze(0)
        pos_c = pos[:, :N].view(-1, K)[:, -1].unsqueeze(0)  # [K-1, 2K-1, ...]

        # Student forward through LoRA-augmented model
        s_out = model(inputs_embeds=compressed, position_ids=pos_c,
                      use_cache=(decompressor is not None))

        student_hidden = s_out.logits  # [B, N/K, V]
        s_logits = student_hidden[:, -1, :]  # Next-token prediction

        # 3. Chunker loss: KL on next-token
        kl_loss = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction="batchmean",
        )

        # 4. Decompressor loss: MSE on full-attention KV
        kv_loss = torch.tensor(0.0, device=device)
        n_kv = 0
        if decompressor is not None and s_out.past_key_values is not None:
            t_kv_out = teacher(input_ids=input_ids, use_cache=True)
            t_cache = t_kv_out.past_key_values
            s_cache = s_out.past_key_values

            for layer_idx in full_attn:
                if layer_idx >= len(s_cache.layers) or layer_idx >= len(t_cache.layers):
                    continue
                s_layer = s_cache.layers[layer_idx]
                t_layer = t_cache.layers[layer_idx]
                if not (hasattr(s_layer, "keys") and hasattr(t_layer, "keys")):
                    continue

                exp_k = decompressor(s_layer.keys)[:, :, :N, :]
                exp_v = decompressor(s_layer.values)[:, :, :N, :]
                kv_loss += F.mse_loss(exp_k, t_layer.keys)
                kv_loss += F.mse_loss(exp_v, t_layer.values)
                n_kv += 1

            if n_kv > 0:
                kv_loss = kv_loss / n_kv

        # Combined
        total_loss = kl_loss + 0.01 * kv_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(chunker.parameters()) + list(decompressor.parameters()) +
            [p for p in model.parameters() if p.requires_grad],
            1.0,
        )
        optimizer.step()
        scheduler.step()

        running_loss += total_loss.item()
        running_kl += kl_loss.item()

        if (step + 1) % 100 == 0:
            avg_loss = running_loss / 100
            avg_kl = running_kl / 100
            pct = 100 * (step + 1) / args.steps
            top1 = (t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean().item()
            print(
                f"  step {step+1:5d}/{args.steps} ({pct:3.0f}%)  "
                f"loss: {avg_loss:.4f}  kl: {avg_kl:.4f}  "
                f"top1: {top1:.3f}  kv({n_kv}): {kv_loss.item():.4f}  "
                f"lr: {scheduler.get_last_lr()[0]:.2e}"
            )
            running_loss = 0.0
            running_kl = 0.0

        if (step + 1) % 500 == 0:
            Path("checkpoints").mkdir(exist_ok=True)
            out = Path("checkpoints") / f"mtc_e2e_k{K}_step{step+1}.pt"
            torch.save({
                "step": step + 1,
                "chunker_state": chunker.state_dict(),
                "decompressor_state": decompressor.state_dict(),
                "lora_state": {
                    k: v for k, v in model.state_dict().items() if "lora" in k
                },
                "chunk_size": K,
                "hidden_dim": dim,
                "kv_head_dim": head_dim,
            }, out)
            print(f"           saved -> {out}")

    print(f"\nDone.")
    print(f"  chunker + decompressor + LoRA saved to checkpoints/")


if __name__ == "__main__":
    main()
