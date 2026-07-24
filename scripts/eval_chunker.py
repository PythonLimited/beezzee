"""Eval a checkpoint on diverse prompts at lengths up to max model context.

Usage:
    python scripts/eval_chunker.py                         # latest checkpoint
    python scripts/eval_chunker.py checkpoints/step_500.pt  # specific checkpoint
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.mtc_model import MTCModel
from src.benchmark import logit_divergence


LENGTHS = [64, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072]

PROMPTS = [
    ("factual",   "The capital of France is Paris. The capital of Germany is Berlin. "),
    ("python",    "def process(data):\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result\n\n"),
    ("narrative", "The old library stood at the end of Elm Street, its windows dark and its "
                  "doors never locked. Nobody remembered who had built it, but everyone "
                  "in town had a story about the books inside. "),
    ("technical", "The transformer model processes input tokens through multiple layers of "
                  "self-attention and feed-forward networks. Each layer applies layer "
                  "normalization before the attention and MLP sublayers. "),
]


def make_prompt(base: str, target_tokens: int, tokenizer) -> str:
    prompt = ""
    while len(tokenizer.encode(prompt)) < target_tokens:
        prompt += base
    return prompt


def main():
    MODEL_DIR = Path("models/Qwen_Qwen3.5-0.8B-Base")

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    ckpt_dir = Path("checkpoints")
    ckpt_files = sorted(ckpt_dir.glob("chunker_*.pt"))
    if not ckpt_files:
        print("No checkpoints found in checkpoints/")
        return
    ckpt_path = ckpt_files[-1]
    if len(sys.argv) > 1:
        ckpt_path = Path(sys.argv[1])

    ckpt = torch.load(ckpt_path, map_location="cpu")
    K = ckpt["chunk_size"]
    print(f"Checkpoint: step={ckpt.get('step', '?')}  chunker={ckpt['chunker_type']}  K={K}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    load_kwargs = dict(dtype=dtype, attn_implementation="eager")
    if device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), **load_kwargs)
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), device_map={"": device}, **load_kwargs
        )
    model.eval()

    mtc = MTCModel(
        base_model=model,
        tokenizer=tokenizer,
        chunk_name=ckpt["chunker_type"],
        chunk_size=K,
    )
    mtc.chunker.load_state_dict(ckpt["chunker_state"])
    mtc.chunker = mtc.chunker.to(device=device, dtype=model.dtype)
    mtc.chunker.eval()

    print(f"\n{'─'*78}")
    print(f"  {'prompt':12s} | {'N':>5s}→{'N/K':>4s} | {'⊿%':>7s} | {'cos':>5s} | {'top1':>4s} | {'std':>7s} | {'mtc':>7s} |  ×")
    print(f"{'─'*78}")

    total, matched = 0, 0

    for label, base in PROMPTS:
        for target in LENGTHS:
            prompt = make_prompt(base, target, tokenizer)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            N = input_ids.shape[1]
            comp = N // K + (1 if N % K else 0)

            try:
                import time

                # Warmup
                with torch.no_grad():
                    _ = mtc.prefill(input_ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                # Time MTC prefill (always works — compressed sequence)
                t0 = time.perf_counter()
                with torch.no_grad():
                    mtc_logits, _ = mtc.prefill(input_ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_mtc = time.perf_counter() - t0

                # Time standard prefill (may OOM at 65K+ with full tokens)
                t_std = 0
                try:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        std_logits, _ = mtc.standard_prefill(input_ids)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_std = time.perf_counter() - t0
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        t_std = 0  # signal: std prefill OOMed
                    else:
                        raise

                speedup = t_std / t_mtc if t_mtc > 0 and t_std > 0 else 0
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  {'':>72s}  ← OOM, skipping {label} at len {target}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise

            if t_std == 0:
                # Standard OOMed — only show MTC time
                std_logits = mtc_logits  # fallback for div comparison
                div = logit_divergence(mtc_logits[0, 0], mtc_logits[0, 0])
                flag = "~"
                speed_str = f"{'N/A':>6s}"
                matched_str = "~"
                print(
                    f"  {flag} {label:11s} | {N:5d}→{comp:4d} | {'---':>8s} | {'---':>5s} |  ~  | {speed_str} | {t_mtc*1000:6.0f}ms | ---"
                )
                continue

            div = logit_divergence(std_logits[0, 0], mtc_logits[0, 0])
            flag = "✓" if div["top1_match"] else " "
            matched += int(div["top1_match"])
            total += 1

            js = div['js_divergence']
            js_pct = js * 100  # % distribution difference
            if js < 1e-5:
                js_str = f"\033[32m{js_pct:.3f}%\033[0m"
            elif js < 1e-4:
                js_str = f"\033[33m{js_pct:.3f}%\033[0m"
            else:
                js_str = f"\033[31m{js_pct:.3f}%\033[0m"

            cos = div['cosine_sim']
            if cos > 0.3:
                cos_str = f"\033[32m{cos:.3f}\033[0m"
            elif cos > 0.1:
                cos_str = f"\033[33m{cos:.3f}\033[0m"
            else:
                cos_str = f"\033[31m{cos:.3f}\033[0m"

            print(
                f"  {flag} {label:11s} | {N:5d}→{comp:4d} | {js_str} | {cos_str} |"
                f"  {flag:>3s} | {t_std*1000:6.0f}ms | {t_mtc*1000:6.0f}ms | ×{speedup:.1f}"
            )

            # Clear MPS memory between long runs
            if device.type == "mps" and N > 4096:
                torch.mps.empty_cache()

    print(f"{'─'*78}")
    print(f"  Top-1 match: {matched}/{total}")
    print(f"{'─'*78}")

    # ── Generation comparison ──
    gen_prompt = "The capital of France is Paris. The capital of Germany is Berlin."
    gen_ids = tokenizer(gen_prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        _, std_cache = mtc.standard_prefill(gen_ids)
        _, mtc_cache = mtc.prefill(gen_ids)

    std_gen = mtc.generate_from_cache(gen_ids, std_cache, max_new_tokens=25)
    mtc_gen = mtc.generate_from_cache(gen_ids, mtc_cache, max_new_tokens=25)

    # Token-level overlap: count matching tokens in first min(len) positions
    min_len = min(std_gen.shape[1], mtc_gen.shape[1])
    token_match = (std_gen[0, :min_len] == mtc_gen[0, :min_len]).sum().item()
    n_tokens = min_len
    first_match = std_gen[0, 0].item() == mtc_gen[0, 0].item() if min_len > 0 else False

    std_text = tokenizer.decode(std_gen[0], skip_special_tokens=True)
    mtc_text = tokenizer.decode(mtc_gen[0], skip_special_tokens=True)

    print(f"\n  ── Generation ──")
    print(f"  std:  {std_text}")
    print(f"  mtc:  {mtc_text}")
    print(f"  tok match:   {token_match}/{n_tokens}  ({100*token_match/max(1,n_tokens):.0f}%)")
    print(f"  1st token:   {'✓' if first_match else '✗'}")


if __name__ == "__main__":
    main()
