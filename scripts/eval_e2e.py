"""End-to-end eval: standard vs MTC prefill + generation.

Usage:
    # Zero-training baseline (mean pool, no decompressor)
    python scripts/eval_e2e.py

    # With trained chunker
    python scripts/eval_e2e.py --chunker checkpoints/chunker_linear_k4_step4000.pt

    # With trained chunker + decompressor
    python scripts/eval_e2e.py --chunker checkpoints/chunker_linear_k4_step4000.pt \\
                               --decompressor checkpoints/decompressor_k4_step5000.pt

    # Sweep specific params
    python scripts/eval_e2e.py --chunker ckpt.pt --decompressor dkpt.pt \\
                               --chunk-sizes 2,4,8 --lengths 256,1024,4096 --gen-tokens 25
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from src.mtc_model import MTCModel
from src.chunkers import build_chunker
from src.kv_decompressor import KVDecompressor
from src.benchmark import logit_divergence

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / MODEL_ID.replace("/", "_")

PROMPT_TEMPLATES = {
    "factual": [
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The largest ocean on Earth is the Pacific Ocean. The Amazon is the longest river. ",
        "In 1969, Neil Armstrong became the first human to walk on the moon. "
        "The Apollo 11 mission was launched from Kennedy Space Center. ",
    ],
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n"
        "        self.left = None\n        self.right = None\n",
    ],
    "narrative": [
        "Once upon a time in a land far away, there lived a brave knight who set out on "
        "a quest to find the legendary golden dragon. The journey was long and perilous. ",
        "The sun was setting over the horizon, casting long shadows across the ancient city. "
        "People hurried through the narrow streets as the evening bells rang. ",
    ],
    "technical": [
        "The transformer architecture uses self-attention mechanisms to process sequential data. "
        "Each token attends to every other token in the sequence with O(N^2) complexity. ",
        "Gradient descent is an iterative optimization algorithm for finding local minima "
        "of differentiable functions. The learning rate controls the step size. ",
    ],
}


def ensure_model_local():
    if not (MODEL_DIR / "config.json").exists():
        print(f"Downloading {MODEL_ID} → {MODEL_DIR} ...")
        snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR), local_dir_use_symlinks=False)


def build_prompt(template: str, target_tokens: int, tokenizer) -> str:
    template_tokens = len(tokenizer.encode(template))
    repeats = max(1, target_tokens // max(1, template_tokens) + 1)
    raw = template * repeats
    encoded = tokenizer.encode(raw)
    if len(encoded) <= target_tokens:
        return raw
    decoded = tokenizer.decode(encoded[:target_tokens], skip_special_tokens=True)
    return decoded


def measure_time(fn, warmup: int = 3, trials: int = 5, device: str = "cpu") -> float:
    for _ in range(warmup):
        fn()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(trials):
        fn()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / trials


def run_standard(model: MTCModel, input_ids: torch.Tensor,
                 max_new_tokens: int = 50, temperature: float = 0.0):
    logits, cache = model.standard_prefill(input_ids)
    output = model.generate_from_cache(input_ids, cache,
                                       max_new_tokens=max_new_tokens,
                                       temperature=temperature)
    return logits, output


def run_mtc(model: MTCModel, input_ids: torch.Tensor,
            max_new_tokens: int = 50, temperature: float = 0.0):
    logits, cache = model.prefill(input_ids)
    output = model.generate_from_cache(input_ids, cache,
                                       max_new_tokens=max_new_tokens,
                                       temperature=temperature)
    return logits, output


def token_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    common = sum(1 for x, y in zip(a.tolist(), b.tolist()) if x == y)
    return common / max(len(a.tolist()), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunker", type=str, default=None)
    parser.add_argument("--decompressor", type=str, default=None)
    parser.add_argument("--chunk-sizes", type=str, default="2,4,8,16",
                        help="Comma-separated chunk sizes")
    parser.add_argument("--lengths", type=str,
                        default="256,512,1024,2048,4096,8192,16384,32768",
                        help="Comma-separated context lengths")
    parser.add_argument("--gen-tokens", type=int, default=25)
    parser.add_argument("--prompt-types", type=str, default="factual,code,narrative,technical",
                        help="Comma-separated prompt types")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Max context length (caps --lengths)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Only compare prefill, skip generation")
    args = parser.parse_args()

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    if args.max_length:
        lengths = [l for l in lengths if l <= args.max_length]
    prompt_types = args.prompt_types.split(",")
    gen_tokens = args.gen_tokens

    # ── Device ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        attn = "flash_attention_2"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        attn = "eager"
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        attn = "eager"

    print(f"\nDevice: {device}  dtype: {dtype}  attn: {attn}\n")

    ensure_model_local()

    # ── Load model ─────────────────────────────────────────────────
    print(f"Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=dtype, attn_implementation=attn,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # ── Load chunker ───────────────────────────────────────────────
    chunk_name = "mean"
    chunker_k = 4
    if args.chunker:
        ckpt = torch.load(args.chunker, map_location="cpu")
        chunk_name = ckpt.get("chunker_type", "linear")
        chunker_k = ckpt.get("chunk_size", 4)
        hidden_dim = ckpt.get("hidden_dim", model.config.hidden_size)
    else:
        hidden_dim = model.config.hidden_size

    # ── Load decompressor ──────────────────────────────────────────
    decompressor = None
    if args.decompressor:
        d_ckpt = torch.load(args.decompressor, map_location="cpu")
        head_dim = d_ckpt.get("head_dim", model.config.head_dim)
        decompressor = KVDecompressor(head_dim, d_ckpt.get("chunk_size", 4), depth=5)
        decompressor.load_state_dict(d_ckpt["decompressor_state"])
        decompressor = decompressor.to(device=device, dtype=dtype).eval()
        print(f"Decompressor loaded: step={d_ckpt.get('step','?')}  "
              f"K={d_ckpt.get('chunk_size','?')}")

    # ── Setup ──────────────────────────────────────────────────────
    total_len = len(prompt_types) * len(lengths) * (1 + len(chunk_sizes))
    print(f"\n{'='*80}")
    print(f"  {'Eval':^76s}")
    print(f"  prompts: {prompt_types}   lengths: {lengths}")
    print(f"  chunk sizes: {chunk_sizes}   gen tokens: {gen_tokens}")
    print(f"  chunker: {args.chunker or '(mean, untrained)'}")
    print(f"  decompressor: {'yes' if decompressor else 'none'}")
    print(f"  samples: {total_len}")
    print(f"{'='*80}")

    mtc = MTCModel(
        base_model=model, tokenizer=tokenizer,
        chunk_name=chunk_name, chunk_size=chunker_k,
        decompressor=decompressor,
    )

    if args.chunker:
        mtc.chunker.load_state_dict(ckpt["chunker_state"])
        mtc.chunker = mtc.chunker.to(device=device, dtype=dtype).eval()

    # ── Print header ───────────────────────────────────────────────
    header = (
        f"{'Type':<10s}  {'Len':>7s}  {'K':>3s}  "
        f"{'StdF(ms)':>9s}  {'MtcF(ms)':>9s}  {'Speedup':>7s}  "
        f"{'StdG(ms)':>9s}  {'MtcG(ms)':>9s}  "
    )
    if not args.no_generate:
        header += f"{'Overlap':>7s}  {'JS_div':>8s}  {'CosSim':>7s}"
    print(f"\n{header}")
    print("-" * len(header))

    for ptype in prompt_types:
        templates = PROMPT_TEMPLATES[ptype]
        for t_len in lengths:
            # Build prompt
            prompt = build_prompt(templates[t_len % len(templates)], t_len, tokenizer)
            input_ids = tokenizer(prompt, return_tensors="pt",
                                  truncation=True, max_length=t_len).input_ids.to(device)
            actual_len = input_ids.shape[1]
            if actual_len < t_len * 0.9:
                continue  # skip if we couldn't build a long-enough prompt

            # ── Standard baseline (once per prompt/length) ─────────
            def _std_prefill():
                return mtc.standard_prefill(input_ids)

            std_prefill_time = measure_time(_std_prefill, trials=3, device=device.type)

            if not args.no_generate:
                def _std_full():
                    return run_standard(mtc, input_ids, max_new_tokens=gen_tokens)

                std_gen_time = measure_time(_std_full, trials=2, device=device.type)
                _, std_output = run_standard(mtc, input_ids, max_new_tokens=gen_tokens)
                std_text = tokenizer.decode(std_output[0], skip_special_tokens=True)
            else:
                std_gen_time = 0.0

            # ── MTC per chunk size ──────────────────────────────────
            for K in chunk_sizes:
                mtc.chunk_size = K

                def _mtc_prefill():
                    return mtc.prefill(input_ids)

                mtc_prefill_time = measure_time(_mtc_prefill, trials=3, device=device.type)
                speedup = std_prefill_time / mtc_prefill_time if mtc_prefill_time > 0 else 0

                if not args.no_generate:
                    def _mtc_full():
                        return run_mtc(mtc, input_ids, max_new_tokens=gen_tokens)

                    mtc_gen_time = measure_time(_mtc_full, trials=2, device=device.type)
                    mtc_logits, mtc_output = run_mtc(mtc, input_ids, max_new_tokens=gen_tokens)
                    mtc_text = tokenizer.decode(mtc_output[0], skip_special_tokens=True)

                    overlap = token_overlap(std_output[0], mtc_output[0])

                    # Divergence on next-token logits
                    std_logits, _ = mtc.standard_prefill(input_ids)
                    mtc_logits, _ = mtc.prefill(input_ids)
                    div = logit_divergence(std_logits[0, 0], mtc_logits[0, 0])
                else:
                    mtc_gen_time = 0.0
                    overlap = 0.0
                    div = {"js_divergence": 0.0, "cosine_sim": 0.0}

                # ── Print row ───────────────────────────────────────
                row = (
                    f"{ptype:<10s}  {actual_len:>7d}  {K:>3d}  "
                    f"{std_prefill_time*1000:>9.1f}  {mtc_prefill_time*1000:>9.1f}  {speedup:>6.1f}x  "
                    f"{std_gen_time*1000:>9.1f}  {mtc_gen_time*1000:>9.1f}  "
                )
                if not args.no_generate:
                    row += (
                        f"{overlap:>7.3f}  {div['js_divergence']:>8.6f}  {div['cosine_sim']:>7.4f}"
                    )
                print(row)

                # Print generation samples for first few
                if not args.no_generate and ptype == prompt_types[0] and K == chunk_sizes[0]:
                    print(f"\n  --- std:  {std_text[:120]}")
                    print(f"  --- mtc:  {mtc_text[:120]}")

            # Spacing
            if not args.no_generate and ptype == prompt_types[0]:
                print()

    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
