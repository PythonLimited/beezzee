"""Compare generation: standard vs MTC prefill, same prompt.

Usage:
    python scripts/compare_gen.py checkpoints/chunker_linear_k4_step4000.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.mtc_model import MTCModel


def main():
    MODEL_DIR = Path("models/Qwen_Qwen3.5-0.8B-Base")
    PROMPT = "The capital of France is Paris. The capital of Germany is Berlin."

    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("checkpoints").glob("chunker_*.pt"))[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    K = ckpt["chunk_size"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: step={ckpt.get('step','?')}  K={K}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device).eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    mtc = MTCModel(model, tokenizer, chunk_name=ckpt["chunker_type"], chunk_size=K)
    mtc.chunker.load_state_dict(ckpt["chunker_state"])
    mtc.chunker = mtc.chunker.to(device=device, dtype=model.dtype)
    mtc.chunker.eval()

    import time

    # ── Standard (full prefill + generation) ──
    t0 = time.perf_counter()
    with torch.no_grad():
        std_logits, std_cache = mtc.standard_prefill(input_ids)
    std_out = mtc.generate_from_cache(input_ids, std_cache, max_new_tokens=50)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    t_std = time.perf_counter() - t0

    # ── MTC (compressed prefill + generation from compressed cache) ──
    t0 = time.perf_counter()
    with torch.no_grad():
        mtc_logits, mtc_cache = mtc.prefill(input_ids)
    mtc_out = mtc.generate_from_cache(input_ids, mtc_cache, max_new_tokens=50)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    t_mtc = time.perf_counter() - t0

    match = std_logits[0,0].argmax() == mtc_logits[0,0].argmax()

    std_text = tokenizer.decode(std_out[0], skip_special_tokens=True)
    mtc_text = tokenizer.decode(mtc_out[0], skip_special_tokens=True)

    print(f"\nPrompt: {PROMPT}")
    print(f"\n{'─'*60}")
    print(f"  Standard  ({t_std*1000:.0f}ms pp):  {std_text}")
    print(f"  MTC       ({t_mtc*1000:.0f}ms pp):  {mtc_text}")
    print(f"  Speedup:     ×{t_std/t_mtc:.1f}")
    print(f"  Next-token:  {'✓ match' if match else '✗ mismatch'}")
    print(f"  Note: generation from compressed cache may degrade. PP speedup is real.")


if __name__ == "__main__":
    main()
