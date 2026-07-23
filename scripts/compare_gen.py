"""Compare generation: standard vs MTC prefill, with optional decompressor.

Usage:
    python scripts/compare_gen.py checkpoints/chunker_linear_k4_step5000.pt
    python scripts/compare_gen.py chunker.pt decompressor.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.kv_decompressor import KVDecompressor
from src.mtc_model import MTCModel


def main():
    MODEL_DIR = Path("models/Qwen_Qwen3.5-0.8B-Base")
    PROMPT = "The capital of France is Paris. The capital of Germany is Berlin."

    if len(sys.argv) < 2:
        ckpts = sorted(Path("checkpoints").glob("chunker_*.pt"))
        if not ckpts:
            print("No chunker checkpoints found")
            return
        chunker_path = ckpts[-1]
    else:
        chunker_path = Path(sys.argv[1])

    decompressor_path = None
    if len(sys.argv) > 2:
        decompressor_path = Path(sys.argv[2])

    ckpt = torch.load(chunker_path, map_location="cpu")
    K = ckpt["chunk_size"]

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()

    decompressor = None
    if decompressor_path and decompressor_path.exists():
        d_ckpt = torch.load(decompressor_path, map_location="cpu")
        decompressor = KVDecompressor(
            d_ckpt.get("kv_head_dim", d_ckpt["hidden_dim"]), d_ckpt["chunk_size"]
        )
        decompressor.load_state_dict(d_ckpt["decompressor_state"])
        decompressor = decompressor.to(device=device, dtype=dtype)
        decompressor.eval()
        print(f"Decompressor: step={d_ckpt['step']}")

    mtc = MTCModel(model, tokenizer, chunk_name=ckpt["chunker_type"],
                   chunk_size=K, decompressor=decompressor)
    mtc.chunker.load_state_dict(ckpt["chunker_state"])
    mtc.chunker = mtc.chunker.to(device=device, dtype=dtype)
    mtc.chunker.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    import time

    with torch.no_grad():
        # Standard
        t0 = time.perf_counter()
        _, std_cache = mtc.standard_prefill(input_ids)
        std_out = mtc.generate_from_cache(input_ids, std_cache, max_new_tokens=25)
        if device.type == "mps": torch.mps.synchronize()
        elif device.type == "cuda": torch.cuda.synchronize()
        t_std = time.perf_counter() - t0

        # MTC (with or without decompressor)
        t0 = time.perf_counter()
        _, mtc_cache = mtc.prefill(input_ids)
        mtc_out = mtc.generate_from_cache(input_ids, mtc_cache, max_new_tokens=25)
        if device.type == "mps": torch.mps.synchronize()
        elif device.type == "cuda": torch.cuda.synchronize()
        t_mtc = time.perf_counter() - t0

    std_text = tokenizer.decode(std_out[0], skip_special_tokens=True)
    mtc_text = tokenizer.decode(mtc_out[0], skip_special_tokens=True)

    print(f"\nPrompt: {PROMPT}")
    print(f"Chunker: step={ckpt.get('step','?')}  K={K}  "
          f"decompressor={'yes' if decompressor else 'no'}")
    print(f"\n  std ({t_std*1000:.0f}ms):  {std_text}")
    print(f"  mtc ({t_mtc*1000:.0f}ms):  {mtc_text}")
    print(f"  speedup: ×{t_std/t_mtc:.1f}")
    print(f"  gen match: {'✓' if std_out.equal(mtc_out) else '✗'}")


if __name__ == "__main__":
    main()
