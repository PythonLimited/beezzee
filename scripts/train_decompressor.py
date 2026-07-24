"""Train KV decompressor from a chunker checkpoint (separate project).

Usage:
    python scripts/train_decompressor.py checkpoints/chunker_linear_k4_step14400.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chunkers import build_chunker
from src.kv_decompressor import KVDecompressor
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
    print(f"Chunker: step={ckpt['step']}  K={K}")

    print(f"Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float16,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
        low_cpu_mem_usage=True,
    ).to(device).eval()

    chunker = build_chunker(ckpt["chunker_type"], ckpt["hidden_dim"], K)
    chunker.load_state_dict(ckpt["chunker_state"])
    chunker = chunker.to(device=device, dtype=model.dtype).eval()
    for p in chunker.parameters():
        p.requires_grad = False

    # Use KV head dim, not hidden_size
    head_dim = model.config.head_dim
    decompressor = KVDecompressor(head_dim, K, depth=5).to(device=device, dtype=model.dtype)
    decompressor.train()

    try:
        layer_types = model.config.layer_types
        full_attention_layers = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    except AttributeError:
        try:
            layer_types = model.config.text_config.layer_types
            full_attention_layers = {i for i, t in enumerate(layer_types) if t == "full_attention"}
        except AttributeError:
            full_attention_layers = set(range(model.config.num_hidden_layers))

    optimizer = torch.optim.AdamW(decompressor.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-5)

    STEPS = 5000
    seq_len = 256
    print(f"\n{'='*60}")
    print(f"  Decompressor: 3-layer MLP, K={K}, head_dim={head_dim}")
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

        B, N_full = input_ids.shape
        N = (N_full // K) * K
        input_ids = input_ids[:, :N]
        dtype = next(model.parameters()).dtype

        # Ground truth: full prefill → KV
        with torch.no_grad():
            gt_out = model(input_ids=input_ids, use_cache=True)
            gt_cache = gt_out.past_key_values

        # Compressed prefill → compressed KV (sparse positions, aligns with teacher)
        with torch.no_grad():
            embeds_full = model.get_input_embeddings()(input_ids)
            compressed = chunker(embeds_full)
            n_full = N // K
            pos = torch.arange(N, device=device).unsqueeze(0)
            pos_c = pos[:, :n_full * K].view(-1, K)[:, -1].unsqueeze(0)
            comp_out = model(inputs_embeds=compressed.to(dtype), position_ids=pos_c, use_cache=True)
            comp_cache = comp_out.past_key_values

        # Decompress and compute MSE for each full-attention layer
        loss = torch.tensor(0.0, device=device)
        n_layers = 0
        for layer_idx, (comp_layer, gt_layer) in enumerate(
            zip(comp_cache.layers, gt_cache.layers)
        ):
            if layer_idx not in full_attention_layers:
                continue
            if not (hasattr(comp_layer, "keys") and hasattr(gt_layer, "keys")):
                continue
            if not hasattr(comp_layer, "values") or not hasattr(gt_layer, "values"):
                continue

            exp_k = decompressor(comp_layer.keys)[:, :, :N, :]
            exp_v = decompressor(comp_layer.values)[:, :, :N, :]
            loss += torch.nn.functional.mse_loss(exp_k, gt_layer.keys)
            loss += torch.nn.functional.mse_loss(exp_v, gt_layer.values)
            n_layers += 1

        if n_layers == 0:
            continue
        loss = loss / n_layers

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decompressor.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1:5d}/{STEPS} | loss: {running_loss/50:.6f} | lr: {scheduler.get_last_lr()[0]:.2e}")
            running_loss = 0.0

        if (step + 1) % 500 == 0:
            Path("checkpoints").mkdir(exist_ok=True)
            out = Path("checkpoints") / f"decompressor_k{K}_step{step+1}.pt"
            torch.save({
                "step": step + 1, "decompressor_state": decompressor.state_dict(),
                "chunk_size": K, "head_dim": head_dim,
            }, out)
            print(f"           saved → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
