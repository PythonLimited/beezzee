"""Quick diagnostic on a chunker checkpoint — one line you can paste to share.

Usage:
    python scripts/checkpoint_report.py checkpoints/chunker_linear_k4_step500.pt
    python scripts/checkpoint_report.py chunker.pt decompressor.pt  # with decompressor
    python scripts/checkpoint_report.py --watch checkpoints/        # monitor directory
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

from src.mtc_model import MTCModel
from src.chunkers import build_chunker
from src.kv_decompressor import KVDecompressor
from src.benchmark import logit_divergence

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / MODEL_ID.replace("/", "_")

PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The largest ocean on Earth is the Pacific Ocean. The Amazon is the longest river. "
    "Gravity is a fundamental force that attracts objects with mass toward each other. "
    "The speed of light in vacuum is approximately 299,792,458 meters per second. "
)

_model = None
_tokenizer = None
_device = None
_dtype = None


def _load_model(device, dtype):
    global _model, _tokenizer, _device, _dtype
    if _model is not None:
        return
    _device = device
    _dtype = dtype
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    _model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=dtype,
        attn_implementation="flash_attention_2" if device.type == "cuda" else "eager",
        low_cpu_mem_usage=True,
    ).to(device).eval()


def report(chunker_path: str, decompressor_path: str | None = None,
           lengths: list[int] | None = None, gen_tokens: int = 25):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    _load_model(device, dtype)

    ckpt = torch.load(chunker_path, map_location="cpu")
    K = ckpt.get("chunk_size", 4)
    chunk_name = ckpt.get("chunker_type", "linear")
    step = ckpt.get("step", "?")
    hidden_dim = ckpt.get("hidden_dim", _model.config.hidden_size)

    decompressor = None
    if decompressor_path and Path(decompressor_path).exists():
        d_ckpt = torch.load(decompressor_path, map_location="cpu")
        d_head = d_ckpt.get("head_dim", d_ckpt.get("kv_head_dim", _model.config.head_dim))
        d_K = d_ckpt.get("chunk_size", K)
        decompressor = KVDecompressor(d_head, d_K, depth=5)
        decompressor.load_state_dict(d_ckpt["decompressor_state"])
        decompressor = decompressor.to(device=device, dtype=dtype).eval()

    mtc = MTCModel(
        base_model=_model, tokenizer=_tokenizer,
        chunk_name=chunk_name, chunk_size=K,
        decompressor=decompressor,
    )
    mtc.chunker.load_state_dict(ckpt["chunker_state"])
    mtc.chunker = mtc.chunker.to(device=device, dtype=dtype).eval()

    if lengths is None:
        lengths = [256, 1024, 4096]

    for L in lengths:
        rep = max(1, L // len(_tokenizer.encode(PROMPT)) + 1)
        prompt = PROMPT * rep
        input_ids = _tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=L).input_ids.to(device)
        actual_len = input_ids.shape[1]
        if actual_len < L * 0.8:
            continue

        # Standard
        t0 = time.perf_counter()
        std_logits, _ = mtc.standard_prefill(input_ids)
        sync(device)
        t_std = time.perf_counter() - t0

        # MTC
        t0 = time.perf_counter()
        mtc_logits, _ = mtc.prefill(input_ids)
        sync(device)
        t_mtc = time.perf_counter() - t0

        div = logit_divergence(std_logits[0, 0], mtc_logits[0, 0])

        # Generation quality (abbreviated)
        _, std_cache = mtc.standard_prefill(input_ids)
        t0 = time.perf_counter()
        std_out = mtc.generate_from_cache(input_ids, std_cache, max_new_tokens=gen_tokens)
        sync(device)
        t_std_gen = time.perf_counter() - t0

        _, mtc_cache = mtc.prefill(input_ids)
        t0 = time.perf_counter()
        mtc_out = mtc.generate_from_cache(input_ids, mtc_cache, max_new_tokens=gen_tokens)
        sync(device)
        t_mtc_gen = time.perf_counter() - t0

        overlap = sum(1 for x, y in zip(std_out[0].tolist(), mtc_out[0].tolist()) if x == y) / len(std_out[0].tolist())
        std_text = _tokenizer.decode(std_out[0], skip_special_tokens=True)
        mtc_text = _tokenizer.decode(mtc_out[0], skip_special_tokens=True)

        speedup = t_std / t_mtc if t_mtc > 0 else 0

        print(
            f"REPORT | step={step:>6s}  K={K}  len={actual_len:>5d}  "
            f"top1={div['top1_match']}  jsd={div['js_divergence']:.4e}  "
            f"cos={div['cosine_sim']:.4f}  "
            f"fillup={speedup:.1f}x({t_std*1000:.0f}/{t_mtc*1000:.0f}ms)  "
            f"gen_ovl={overlap:.2f}  "
            f"gen_up={t_std_gen/t_mtc_gen if t_mtc_gen>0 else 0:.1f}x"
            f"  {'[DECOMP]' if decompressor else ''}"
        )
        print(f"REPORT |   std: {std_text[:80]}")
        print(f"REPORT |   mtc: {mtc_text[:80]}")
        break  # just longest length that fits


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def watch(checkpoints_dir: str, decompressor_path: str | None = None,
          poll_interval: float = 30):
    """Poll for new checkpoints and report on each."""
    ckpt_dir = Path(checkpoints_dir)
    seen = set()
    print(f"Watching {ckpt_dir} for new chunker checkpoints (interval={poll_interval}s)...")
    print("REPORT format: step  K  len  top1  jsd  cos  speedup  gen_overlap  gen_output")
    print()

    while True:
        ckpts = sorted(ckpt_dir.glob("chunker_*.pt"))
        for ckpt_path in ckpts:
            if ckpt_path in seen:
                continue
            seen.add(ckpt_path)
            time.sleep(2)  # let file finish writing
            try:
                report(str(ckpt_path), decompressor_path)
            except Exception as e:
                print(f"ERROR on {ckpt_path}: {e}")
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunker", type=str, nargs="?", default=None,
                        help="Path to chunker checkpoint")
    parser.add_argument("decompressor", type=str, nargs="?", default=None,
                        help="Path to decompressor checkpoint (optional)")
    parser.add_argument("--watch", type=str, default=None,
                        help="Watch a checkpoints directory for new chunker files")
    parser.add_argument("--lengths", type=str, default="256,1024,4096",
                        help="Comma-separated context lengths to test")
    parser.add_argument("--gen-tokens", type=int, default=25)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]

    if args.watch:
        watch(args.watch, args.decompressor, poll_interval=10)
    elif args.chunker:
        report(args.chunker, args.decompressor, lengths=lengths, gen_tokens=args.gen_tokens)
    else:
        print("Usage: checkpoint_report.py CHUNKER [DECOMPRESSOR]")
        print("   or: checkpoint_report.py --watch checkpoints/")


if __name__ == "__main__":
    main()
