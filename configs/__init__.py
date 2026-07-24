"""Training configuration — tweak these values for different runs."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # ── Model ──
    model_id: str = "Qwen/Qwen3.5-0.8B-Base"
    attn_implementation: str = "eager"   # eager | sdpa | flash_attention_2
    dtype: str = "float16"

    # ── Chunker ──
    chunker_type: str = "linear"         # mean | attention | linear
    chunk_size: int = 4                  # K tokens → 1 embedding

    # ── Decompressor ──
    train_decompressor: bool = False     # enable after chunker converges

    # ── Training ──
    steps: int = 14000
    eval_every: int = 100
    kl_temperature: float = 2.0
    lr: float = 2e-4  # higher LR for faster convergence
    weight_decay: float = 0.01
    lr_scheduler_tmax: int = 15000
    grad_clip: float = 1.0

    # ── Data ──
    length_schedule: dict[int, int] = field(default_factory=lambda: {
        0: 256, 2000: 512, 4000: 1024, 6000: 2048,
        8000: 4096, 10000: 8192, 11500: 16384, 13000: 32768,
        14000: 65536, 14500: 131072,
    })

    # ── Paths ──
    models_dir: Path = field(default_factory=lambda: Path("models"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    @property
    def local_model_dir(self) -> Path:
        return self.models_dir / self.model_id.replace("/", "_")


# ── Presets ────────────────────────────────────────────────────────

# Fast iteration on MPS
qwen3_5_08b_mps = TrainConfig(
    model_id="Qwen/Qwen3.5-0.8B-Base",
    dtype="float16",
    steps=5000,
    length_schedule={
        0: 256, 500: 512, 1000: 1024, 1500: 2048,
        2000: 4096, 2500: 8192, 3000: 16384, 3500: 32768,
        4000: 65536, 4500: 131072,
    },
)

# Full-scale on a GPU
qwen3_6_27b_gpu = TrainConfig(
    model_id="Qwen/Qwen3.6-27B-FP8",
    dtype="bfloat16",
    attn_implementation="flash_attention_2",
    steps=14000,
    chunk_size=4,
    length_schedule={
        0: 256, 2000: 512, 4000: 1024, 6000: 2048,
        8000: 4096, 10000: 8192, 11500: 16384, 13000: 32768,
        14000: 65536, 14500: 131072,
    },
)

# Fast iteration on DGX A100
qwen3_5_08b_dgx = TrainConfig(
    model_id="Qwen/Qwen3.5-0.8B-Base",
    dtype="bfloat16",
    attn_implementation="flash_attention_2",
    steps=14500,
    chunk_size=4,
    length_schedule={
        0: 256, 2000: 512, 4000: 1024, 6000: 2048,
        8000: 4096, 10000: 8192, 11500: 16384, 13000: 32768,
        14000: 65536, 14500: 131072,
    },
)
