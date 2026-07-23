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


LENGTHS = [64, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
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
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
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

    print(f"\n{'─'*72}")
    print(f"  {'prompt':12s} {'tokens':>6s} {'→comp':>6s}  {'JS div':>10s}  {'cos':>6s}  top1")
    print(f"{'─'*72}")

    total, matched = 0, 0

    for label, base in PROMPTS:
        for target in LENGTHS:
            prompt = make_prompt(base, target, tokenizer)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            N = input_ids.shape[1]
            comp = N // K + (1 if N % K else 0)

            try:
                with torch.no_grad():
                    std_logits, _ = mtc.standard_prefill(input_ids)
                    mtc_logits, _ = mtc.prefill(input_ids)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  {'':>72s}  ← OOM, skipping {label} at len {target}")
                    break
                raise

            div = logit_divergence(std_logits[0, 0], mtc_logits[0, 0])
            flag = "✓" if div["top1_match"] else " "
            matched += int(div["top1_match"])
            total += 1

            print(
                f"  {flag} {label:11s} {N:6d} {comp:6d}  "
                f"{div['js_divergence']:10.2e}  {div['cosine_sim']:6.4f}"
            )

            # Clear MPS memory between long runs
            if device.type == "mps" and N > 4096:
                torch.mps.empty_cache()

    print(f"{'─'*72}")
    print(f"  Top-1 match: {matched}/{total}")
    print(f"{'─'*72}")


if __name__ == "__main__":
    main()
