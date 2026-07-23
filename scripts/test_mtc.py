"""Quick end-to-end test of MTC vs standard prefill."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from src.mtc_model import MTCModel
from src.benchmark import benchmark, logit_divergence


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / MODEL_ID.replace("/", "_")


def ensure_model_local():
    if not (MODEL_DIR / "config.json").exists():
        print(f"Downloading {MODEL_ID} → {MODEL_DIR} ...")
        snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR), local_dir_use_symlinks=False)
    else:
        print(f"Model already at {MODEL_DIR}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    ensure_model_local()

    print(f"Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    print("Model ready.")

    PROMPTS = [
        "The capital of France is Paris. The capital of Germany is Berlin. "
        * 20,
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    "
        * 25,
    ]

    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'='*60}")
        print(f"Prompt {i+1}: \"{prompt[:80]}...\"  ({len(tokenizer.encode(prompt))} tokens)")
        print(f"{'='*60}")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        mtc = MTCModel(
            base_model=model,
            tokenizer=tokenizer,
            chunk_name="mean",
            chunk_size=4,
        )

        print("\n--- Standard prefill ---")
        with torch.no_grad():
            std_logits, std_cache = mtc.standard_prefill(input_ids)
        std_topk = std_logits[0, 0].topk(5)
        print(f"  top-5 tokens: {[tokenizer.decode(t) for t in std_topk.indices]}")
        print(f"  KV cache length: {std_cache.get_seq_length()}")

        print("\n--- MTC (mean chunk=4) ---")
        with torch.no_grad():
            mtc_logits, mtc_cache = mtc.prefill(input_ids)
        mtc_topk = mtc_logits[0, 0].topk(5)
        print(f"  top-5 tokens: {[tokenizer.decode(t) for t in mtc_topk.indices]}")
        print(f"  KV cache length: {mtc_cache.get_seq_length()}")

        div = logit_divergence(std_logits[0, 0], mtc_logits[0, 0])
        print(f"\n  Top-1 match: {div['top1_match']}")
        print(f"  JS divergence: {div['js_divergence']:.6f}")
        print(f"  Cosine sim: {div['cosine_sim']:.6f}")

        print("\n--- Speed benchmark ---")
        results = benchmark(mtc, input_ids, chunk_sizes=[2, 4, 8, 16], trial_tokens=10)
        std_time = results["standard"]["time_per_trial"]
        print(f"  Standard prefill: {std_time*1000:.1f}ms")
        for lbl, data in sorted(results["mtc"].items()):
            mtc_time = data["prefill_time"]
            speedup = std_time / mtc_time
            print(f"  MTC {lbl:>10}: {mtc_time*1000:>8.1f}ms  (speedup: {speedup:.1f}x)")


if __name__ == "__main__":
    main()
