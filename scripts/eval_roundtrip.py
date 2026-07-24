"""Roundtrip eval: standard vs MTC+decompressor generation quality.

Usage:
    python scripts/eval_roundtrip.py checkpoints/chunker_linear_k4_step14400.pt checkpoints/decompressor_k4_step5000.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.kv_decompressor import KVDecompressor
from src.mtc_model import MTCModel


PROMPTS = [
    "The capital of France is Paris. The capital of Germany is Berlin.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "The transformer architecture revolutionized natural language processing by",
]


def main():
    MODEL_DIR = Path("models/Qwen_Qwen3.5-0.8B-Base")

    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_roundtrip.py <chunker.pt> [decompressor.pt]")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    # Load chunker
    ckpt = torch.load(sys.argv[1], map_location="cpu")
    K = ckpt["chunk_size"]

    # Load decompressor (optional)
    decompressor = None
    if len(sys.argv) > 2 and Path(sys.argv[2]).exists():
        d_ckpt = torch.load(sys.argv[2], map_location="cpu")
        decompressor = KVDecompressor(d_ckpt["head_dim"], d_ckpt["chunk_size"])
        decompressor.load_state_dict(d_ckpt["decompressor_state"])
        decompressor = decompressor.to(device=device, dtype=dtype).eval()
        print(f"Decompressor: step={d_ckpt['step']}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=dtype, attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()

    mtc = MTCModel(model, tokenizer, chunk_name=ckpt["chunker_type"],
                   chunk_size=K, decompressor=decompressor)
    mtc.chunker.load_state_dict(ckpt["chunker_state"])
    mtc.chunker = mtc.chunker.to(device=device, dtype=dtype).eval()

    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            # Standard
            _, std_cache = mtc.standard_prefill(ids)
            std_out = mtc.generate_from_cache(ids, std_cache, max_new_tokens=30)

            # MTC + decompressor
            _, mtc_cache = mtc.prefill(ids)
            mtc_out = mtc.generate_from_cache(ids, mtc_cache, max_new_tokens=30)

        std_text = tokenizer.decode(std_out[0], skip_special_tokens=True)
        mtc_text = tokenizer.decode(mtc_out[0], skip_special_tokens=True)

        # Token overlap
        min_len = min(std_out.shape[1], mtc_out.shape[1])
        overlap = (std_out[0, :min_len] == mtc_out[0, :min_len]).sum().item()

        print(f"\n{'─'*70}")
        print(f"  Prompt:  {prompt[:60]}...")
        print(f"  std:     {std_text}")
        print(f"  mtc:     {mtc_text}")
        print(f"  overlap: {overlap}/{min_len} ({100*overlap/max(1,min_len):.0f}%)")
        print(f"{'─'*70}")


if __name__ == "__main__":
    main()
