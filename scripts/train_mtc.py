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
from src.kv_decompressor import KVDecompressor
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
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint .pt file")
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

    decompressor = None
    if cfg.train_decompressor:
        decompressor = KVDecompressor(
            model.config.head_dim, cfg.chunk_size  # KV head dim, not hidden_size
        )
        decompressor = decompressor.to(device=device, dtype=model.dtype)
        decompressor.train()

    trainer = MTCTrainer(
        base_model=model, chunker=chunker, chunk_size=cfg.chunk_size,
        decompressor=decompressor,
    )
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        chunker.load_state_dict(ckpt["chunker_state"])
        start_step = ckpt.get("step", 0)
        if "optimizer_state" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            trainer.scheduler.load_state_dict(ckpt["scheduler_state"])
        if cfg.train_decompressor and decompressor is not None and "decompressor_state" in ckpt:
            decompressor.load_state_dict(ckpt["decompressor_state"])
        if accelerator.is_main_process:
            print(f"Resumed from step {start_step}")

    trainer.optimizer.param_groups[0]["lr"] = cfg.lr
    trainer.optimizer.param_groups[0]["weight_decay"] = cfg.weight_decay
    cfg.checkpoints_dir.mkdir(exist_ok=True)

    # Pool: all lengths up to the current milestone (anti-forgetting + memory-safe)
    import random
    all_lengths = sorted(set(cfg.length_schedule.values()))
    length_milestones = sorted(cfg.length_schedule.items())  # [(step, max_length), ...]
    milestone_idx = 0
    available_lengths = [all_lengths[0]]

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

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"  Chunker:   {cfg.chunker_type}, K={cfg.chunk_size}")
        print(f"  Params:    {sum(p.numel() for p in chunker.parameters()):,}")
        print(f"  Steps:     {cfg.steps}")
        print(f"  Lengths:   {all_lengths}")

    running_loss = 0.0
    running_kv = 0.0
    running_top1 = 0.0

    baseline_loss = None  # set after first 100 steps

    if args.profile:
        import time
        t_data = t_step = t_eval = 0.0
        t_count = 0
        t_wall_start = time.perf_counter()

    # Print column header once
    if accelerator.is_main_process:
        print()
        print(f"{'─'*85}")
        print(f"  {'step':>5s}  {'done%':>5s}  {'Δ_loss':>8s}  {'top1%':>5s}  "
              f"{'quality':>9s}  {'lr':>7s}  {'len':>5s}  {'phase':>8s}")
        print(f"  {'─'*5}  {'─'*5}  {'─'*8}  {'─'*5}  "
              f"{'─'*9}  {'─'*7}  {'─'*5}  {'─'*8}")
        print()

    for step in range(start_step, cfg.steps):
        # Expand available lengths at milestone steps
        while (milestone_idx < len(length_milestones) and
               step >= length_milestones[milestone_idx][0]):
            max_len = length_milestones[milestone_idx][1]
            available_lengths = [l for l in all_lengths if l <= max_len]
            if accelerator.is_main_process and milestone_idx > 0:
                print(f"\n  ══ +{max_len} tokens (now {len(available_lengths)} lengths, step {step}) ══\n")
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

                pct_done = 100 * (step + 1) / cfg.steps
                pct_top1 = 100 * avg_top1

                # Track baseline (mean-pool quality floor) for loss improvement
                if baseline_loss is None:
                    baseline_loss = avg_loss

                # Phase label
                lengths_now = sorted(available_lengths)
                if len(lengths_now) <= 2:
                    phase = "warmup"
                elif max(lengths_now) <= 2048:
                    phase = "short"
                elif max(lengths_now) <= 16384:
                    phase = "mid"
                else:
                    phase = "long"

                # Quality label
                if pct_top1 >= 70:
                    quality = "GREAT"
                elif pct_top1 >= 40:
                    quality = "GOOD"
                elif pct_top1 >= 15:
                    quality = "OK"
                elif pct_top1 >= 5:
                    quality = "learning"
                else:
                    quality = "noise"

                loss_delta = (baseline_loss - avg_loss) / baseline_loss * 100 if baseline_loss > 0 else 0
                delta_str = f"-{loss_delta:.1f}%" if loss_delta > 0.1 else "  ~flat"

                print(
                    f"  {step+1:5d}  {pct_done:4.0f}%  "
                    f"{delta_str:>8s}  {pct_top1:4.0f}%  "
                    f"{quality:>9s}  {metrics['lr']:5.0e}  "
                    f"{input_ids.shape[1]:5d}  {phase:>8s}"
                )

                # Eval mini-report
                eval_prompts = [
                    ("The history of artificial intelligence dates back to the 1950s when "
                     "researchers first began exploring the possibility of machine reasoning. "
                     "Early systems used symbolic logic and rule-based approaches to solve "),
                    ("In computer science, data structures are specialized formats for "
                     "organizing and storing data. Common types include arrays, linked lists, "
                     "trees, and hash tables, each with distinct performance characteristics. "),
                    ("The solar system consists of the Sun and the objects that orbit it, "
                     "including eight planets, dwarf planets, moons, and countless asteroids. "
                     "Jupiter is the largest planet with a mass greater than all others combined. "),
                ]
                eval_text = eval_prompts[(step + 1) % len(eval_prompts)]
                eval_ids = tokenizer(
                    eval_text * 8,
                    return_tensors="pt",
                    truncation=True, max_length=512,
                ).input_ids.to(device)

                if args.profile:
                    t0_e = time.perf_counter()

                eval_metrics_res = eval_step(trainer, eval_ids)

                if args.profile:
                    t_eval += time.perf_counter() - t0_e

                e_top1 = 100 * eval_metrics_res['eval_top1']
                e_qual = "++" if e_top1 >= 50 else " ." if e_top1 >= 10 else ".."

                print(
                    f"           eval| "
                    f"loss: {eval_metrics_res['eval_loss']:.5f}  "
                    f"top1: {e_top1:.0f}%  "
                    f"{e_qual}"
                )

                running_loss = 0.0
                running_top1 = 0.0

                ckpt_path = (
                    cfg.checkpoints_dir
                    / f"chunker_{cfg.chunker_type}_k{cfg.chunk_size}_step{step+1}.pt"
                )
                save_dict = {
                    "step": step + 1,
                    "chunker_state": accelerator.unwrap_model(chunker).state_dict(),
                    "optimizer_state": trainer.optimizer.state_dict(),
                    "scheduler_state": trainer.scheduler.state_dict(),
                    "chunker_type": cfg.chunker_type,
                    "chunk_size": cfg.chunk_size,
                    "hidden_dim": model.config.hidden_size,
                    "kv_head_dim": model.config.head_dim,
                    "metrics": metrics,
                }
                if cfg.train_decompressor:
                    save_dict["decompressor_state"] = accelerator.unwrap_model(decompressor).state_dict()
                torch.save(save_dict, ckpt_path)
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
