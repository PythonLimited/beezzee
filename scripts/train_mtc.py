"""Train the MTC chunker via self-distillation.

Single GPU:
    python scripts/train_mtc.py --config qwen3_5_08b_dgx

Multi-GPU (data parallel):
    accelerate launch --config_file configs/accelerate_dgx.yaml scripts/train_mtc.py --config qwen3_5_08b_dgx
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from configs import TrainConfig, qwen3_5_08b_mps, qwen3_6_27b_gpu, qwen3_5_08b_dgx
from src.chunkers import build_chunker
from src.trainer import MTCTrainer, TextDataset, eval_step

PRESETS = {
    "qwen3_5_08b_mps": qwen3_5_08b_mps,
    "qwen3_6_27b_gpu": qwen3_6_27b_gpu,
    "qwen3_5_08b_dgx": qwen3_5_08b_dgx,
}


def ensure_model_local(cfg: TrainConfig, accelerator):
    local = cfg.local_model_dir
    done_file = local / ".download_complete"
    if done_file.exists():
        return

    if accelerator.is_main_process:
        print(f"Downloading {cfg.model_id} → {local} ...")
        snapshot_download(cfg.model_id, local_dir=str(local), local_dir_use_symlinks=False)
        done_file.touch()

    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, choices=list(PRESETS))
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    cfg = PRESETS.get(args.config, TrainConfig())

    accelerator = Accelerator(mixed_precision="bf16" if cfg.dtype == "bfloat16" else "no")
    # Accelerator only knows CUDA/CPU — handle MPS manually
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = accelerator.device

    if accelerator.is_main_process:
        print(f"Device: {device}")
        print(f"GPUs:   {accelerator.num_processes}")
        print(f"Model:  {cfg.model_id}")
        print(f"Config: {args.config or 'default'}")

    ensure_model_local(cfg, accelerator)

    if accelerator.is_main_process:
        print(f"Loading model from {cfg.local_model_dir} ...")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.local_model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.local_model_dir),
        dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    chunker = build_chunker(cfg.chunker_type, model.config.hidden_size, cfg.chunk_size)
    # fp32 chunker on MPS (fp16 gradients unstable), match model dtype on CUDA
    chunk_dtype = torch.float32 if device.type == "mps" else model.dtype
    chunker = chunker.to(device=device, dtype=chunk_dtype)
    chunker.train()

    trainer = MTCTrainer(base_model=model, chunker=chunker, chunk_size=cfg.chunk_size)
    trainer.optimizer.param_groups[0]["lr"] = cfg.lr
    trainer.optimizer.param_groups[0]["weight_decay"] = cfg.weight_decay
    trainer.scheduler.T_max = cfg.lr_scheduler_tmax
    cfg.checkpoints_dir.mkdir(exist_ok=True)

    # Dataset pool: randomly sample from all seen lengths to prevent forgetting
    import random
    pool_lengths = sorted(cfg.length_schedule.values())
    datasets: dict[int, TextDataset] = {}
    data_iters: dict[int, any] = {}

    def get_iter(seq_len):
        if seq_len not in datasets:
            datasets[seq_len] = TextDataset(
                tokenizer, seq_len=seq_len,
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
            )
        if seq_len not in data_iters:
            data_iters[seq_len] = iter(datasets[seq_len])
        return data_iters[seq_len]

    # Track which lengths are available at each step
    length_milestones = sorted(cfg.length_schedule.items())  # [(step, length), ...]
    available_lengths = [pool_lengths[0]]  # start with smallest

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"  Chunker:   {cfg.chunker_type}, K={cfg.chunk_size}")
        print(f"  Params:    {sum(p.numel() for p in chunker.parameters()):,}")
        print(f"  Steps:     {cfg.steps}")
        print(f"  Lengths:   {pool_lengths} (added progressively)")
        print(f"  Sampling:  random from available lengths")
        print(f"{'='*60}\n")

    running_loss = 0.0
    running_top1 = 0.0
    milestone_idx = 0

    if args.profile:
        import time
        t_data = t_step = t_eval = 0.0
        t_count = 0
        t_wall_start = time.perf_counter()

    for step in range(cfg.steps):
        # Add new lengths to the pool at milestone steps
        while (milestone_idx < len(length_milestones) and
               step >= length_milestones[milestone_idx][0]):
            new_len = length_milestones[milestone_idx][1]
            if new_len not in available_lengths:
                available_lengths.append(new_len)
                if accelerator.is_main_process:
                    print(f"\n  ══ +{new_len} tokens (now {len(available_lengths)} lengths in pool, step {step}) ══\n")
            milestone_idx += 1

        # Randomly pick a length from available pool
        seq_len = random.choice(available_lengths)
        data_iter = get_iter(seq_len)

        if args.profile:
            t0 = time.perf_counter()

        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            data_iters.pop(seq_len, None)
            data_iter = get_iter(seq_len)
            input_ids = next(data_iter).to(device)

        if args.profile:
            t_data += time.perf_counter() - t0
            t0 = time.perf_counter()

        metrics = trainer.train_step(input_ids)

        trainer.optimizer.zero_grad()
        accelerator.backward(metrics["_proxy_loss"])
        if accelerator.sync_gradients:
            torch.nn.utils.clip_grad_norm_(chunker.parameters(), 1.0)
        trainer.optimizer.step()
        trainer.scheduler.step()

        if args.profile:
            t_step += time.perf_counter() - t0

        running_loss += metrics["loss"]
        running_top1 += metrics["top1_match"]

        if (step + 1) % cfg.eval_every == 0:
            if accelerator.is_main_process:
                if args.profile:
                    t0 = time.perf_counter()
                    t_count += 1

                avg_loss = running_loss / cfg.eval_every
                avg_top1 = running_top1 / cfg.eval_every
                print(
                    f"  Step {step+1:5d}/{cfg.steps} | "
                    f"loss: {avg_loss:.6f} | "
                    f"top1: {avg_top1:.3f} | "
                    f"lr: {metrics['lr']:.4e}"
                )

                running_loss = 0.0
                running_top1 = 0.0

                eval_ids = tokenizer(
                    "The capital of France is Paris. " * 8,
                    return_tensors="pt",
                ).input_ids.to(device)

                if args.profile:
                    t0_e = time.perf_counter()

                eval_metrics = eval_step(trainer, eval_ids)

                if args.profile:
                    t_eval += time.perf_counter() - t0_e

                print(
                    f"           eval | "
                    f"loss: {eval_metrics['eval_loss']:.6f} | "
                    f"top1: {eval_metrics['eval_top1']:.3f}"
                )

                ckpt_path = (
                    cfg.checkpoints_dir
                    / f"chunker_{cfg.chunker_type}_k{cfg.chunk_size}_step{step+1}.pt"
                )
                torch.save(
                    {
                        "step": step + 1,
                        "chunker_state": accelerator.unwrap_model(chunker).state_dict(),
                        "chunker_type": cfg.chunker_type,
                        "chunk_size": cfg.chunk_size,
                        "hidden_dim": model.config.hidden_size,
                        "metrics": metrics,
                    },
                    ckpt_path,
                )
                print(f"           saved → {ckpt_path}")

                if args.profile and t_count > 0:
                    elapsed = time.perf_counter() - t_wall_start
                    steps_done = step + 1
                    steps_left = cfg.steps - steps_done
                    sec_per_step = elapsed / steps_done
                    eta = sec_per_step * steps_left
                    print(
                        f"           ── elapsed: {elapsed/60:5.1f}m  "
                        f"eta: {eta/60:5.1f}m  "
                        f"step: {t_step/t_count*1000:6.0f}ms/block"
                        f"  ({t_step/t_count/cfg.eval_every*1000:.1f}ms/step)"
                    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"\nDone. Final checkpoint in {cfg.checkpoints_dir}/")


if __name__ == "__main__":
    main()
