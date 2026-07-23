# MTC — Multi-Token Consumption

Prompt processing speedup: compress N tokens → N/K before the model, preserving next-token quality.

## Architecture

```
Standard:  [N tokens] → Embed → [24 layers] → [logits]         O(N²) attention

MTC:       [N tokens] → Embed → [Chunker K:1] → [24 layers]    O((N/K)²) attention
                                 4.2M trained params    on N/K tokens
```

The chunker is trained via **self-distillation** — KL divergence between the
full model's next-token logits and the compressed forward's logits. No labels needed.

## Quick results (Qwen3.5-0.8B, MPS, zero-training mean pool)

| Chunk | Prefill | Speedup | Top-1 Match | JS Div |
|-------|---------|---------|-------------|--------|
| —     | 179ms   | 1.0x    | —           | —      |
| 4     | 104ms   | 1.7x    | True        | 4e-6   |
| 8     | 68ms    | 2.6x    | —           | —      |
| 16    | 68ms    | 2.6x    | —           | —      |

## Training

```bash
source .venv/bin/activate
python scripts/train_mtc.py
```

Config in the script:
- Model: `Qwen/Qwen3.5-0.8B-Base` (swap to `Qwen/Qwen3.6-27B-FP8`)
- Progressive seq len: 256 → 512 → 1024 → 2048 → 4096
- Chunker: `linear` (4.2M params at 1024-dim, ~105M at 5120-dim)
- Checkpoints saved to `checkpoints/`

## Files

```
src/
  chunkers.py     — MeanChunk, AttnChunk, LinearChunk
  mtc_model.py    — MTCModel wrapper (prefill + generate)
  trainer.py      — Self-distillation training + data
  benchmark.py    — Speed + quality measurement
scripts/
  test_mtc.py     — Inference benchmark
  train_mtc.py    — Training loop
```
