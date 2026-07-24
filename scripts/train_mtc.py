"""Train the MTC chunker via self-distillation.

Single GPU:
    python scripts/train_mtc.py --config qwen3_5_08b_dgx

Multi-GPU:
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


def ensure_model_local(cfg, accelerator):
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
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = PRESETS.get(args.config, TrainConfig())

    accelerator = Accelerator(mixed_precision="bf16" if cfg.dtype == "bfloat16" else "no")
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
    chunk_dtype = torch.float32 if device.type == "mps" else model.dtype
    chunker = chunker.to(device=device, dtype=chunk_dtype)
    chunker.train()

    decompressor = None
    if cfg.train_decompressor:
        decompressor = KVDecompressor(model.config.head_dim, cfg.chunk_size)
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
        if cfg.train_decompressor and decompressor and "decompressor_state" in ckpt:
            decompressor.load_state_dict(ckpt["decompressor_state"])
        if accelerator.is_main_process:
            print(f"Resumed from step {start_step}")

    trainer.optimizer.param_groups[0]["lr"] = cfg.lr
    trainer.optimizer.param_groups[0]["weight_decay"] = cfg.weight_decay
    trainer.scheduler.T_max = cfg.lr_scheduler_tmax
    cfg.checkpoints_dir.mkdir(exist_ok=True)

    trainer.optimizer, trainer.scheduler = accelerator.prepare(trainer.optimizer, trainer.scheduler)

    # ── Progressive training: one length at a time ──
    seq_len = cfg.length_schedule[0]
    dataset = TextDataset(tokenizer, seq_len=seq_len,
                          rank=accelerator.process_index,
                          world_size=accelerator.num_processes)
    data_iter = iter(dataset)

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"  Chunker:   {cfg.chunker_type}, K={cfg.chunk_size}")
        print(f"  Params:    {sum(p.numel() for p in chunker.parameters()):,}")
        print(f"  Steps:     {cfg.steps}")
        print(f"  Schedule:  progressive {dict(sorted(cfg.length_schedule.items()))}")
        print(f"{'='*60}\n")

    running_loss = 0.0
    running_kv = 0.0
    running_top1 = 0.0

    if args.profile:
        import time
        t_step = 0.0
        t_count = 0
        t_wall_start = time.perf_counter()

    for step in range(start_step, cfg.steps):
        if step > 0 and step in cfg.length_schedule:
            new_len = cfg.length_schedule[step]
            if accelerator.is_main_process:
                print(f"\n  ══ Seq len {seq_len} → {new_len}  (step {step}) ══\n")
            seq_len = new_len
            dataset = TextDataset(tokenizer, seq_len=new_len,
                                  rank=accelerator.process_index,
                                  world_size=accelerator.num_processes)
            data_iter = iter(dataset)

        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            dataset = TextDataset(tokenizer, seq_len=seq_len,
                                  rank=accelerator.process_index,
                                  world_size=accelerator.num_processes)
            data_iter = iter(dataset)
            input_ids = next(data_iter).to(device)

        if args.profile:
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
        running_kv += metrics.get("loss_kv", 0)
        running_top1 += metrics["top1_match"]

        if (step + 1) % cfg.eval_every == 0:
            if accelerator.is_main_process:
                avg_loss = running_loss / cfg.eval_every
                avg_kv = running_kv / cfg.eval_every
                avg_top1 = running_top1 / cfg.eval_every
                kv_str = f" | kv:{avg_kv:.6f}" if cfg.train_decompressor else ""

                print(
                    f"  Step {step+1:5d}/{cfg.steps} | "
                    f"loss: {avg_loss:.6f}{kv_str} | "
                    f"top1: {avg_top1:.3f} | "
                    f"lr: {metrics['lr']:.4e}"
                )

                running_loss = 0.0
                running_kv = 0.0
                running_top1 = 0.0

                eval_ids = tokenizer("The capital of France is Paris. " * 8, return_tensors="pt").input_ids.to(device)
                eval_metrics = eval_step(trainer, eval_ids)
                print(f"           eval | loss: {eval_metrics['eval_loss']:.6f} | top1: {eval_metrics['eval_top1']:.3f}")

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

                ckpt_path = cfg.checkpoints_dir / f"chunker_{cfg.chunker_type}_k{cfg.chunk_size}_step{step+1}.pt"
                torch.save(save_dict, ckpt_path)
                print(f"           saved → {ckpt_path}")

                if args.profile and t_count > 0:
                    elapsed = time.perf_counter() - t_wall_start
                    steps_done = step + 1 - start_step
                    steps_left = cfg.steps - step - 1
                    sec_per_step = elapsed / max(steps_done, 1)
                    eta = sec_per_step * steps_left
                    print(f"           ── elapsed: {elapsed/60:5.1f}m  eta: {eta/60:5.1f}m  step: {t_step/t_count*1000:6.0f}ms/block")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"\nDone. Final checkpoint in {cfg.checkpoints_dir}/")


if __name__ == "__main__":
    main()
