"""Benchmark and comparison utilities for MTC."""

import time
import torch
import torch.nn.functional as F
from src.mtc_model import MTCModel


def measure_time(fn, warmup: int = 2, trials: int = 10) -> float:
    for _ in range(warmup):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(trials):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / trials


def benchmark(
    mtc: MTCModel,
    input_ids: torch.Tensor,
    chunk_sizes: list[int] = [2, 4, 8],
    trial_tokens: int = 20,
) -> dict:
    device = input_ids.device

    results = {"standard": {}, "mtc": {}}

    def run_standard():
        return mtc.standard_prefill(input_ids)

    results["standard"]["time_per_trial"] = measure_time(run_standard, trials=5)

    def gen_standard():
        logits, cache = mtc.standard_prefill(input_ids)
        return mtc.generate_from_cache(input_ids, cache, max_new_tokens=trial_tokens)

    results["standard"]["gen_time"] = measure_time(gen_standard, trials=3)

    for k in chunk_sizes:
        mtc.chunk_size = k
        lbl = f"chunk_{k}"

        def run_mtc():
            return mtc.prefill(input_ids)

        results["mtc"][lbl] = {}
        results["mtc"][lbl]["prefill_time"] = measure_time(run_mtc, trials=5)

        def gen_mtc():
            logits, cache = mtc.prefill(input_ids)
            return mtc.generate_from_cache(input_ids, cache, max_new_tokens=trial_tokens)

        results["mtc"][lbl]["gen_time"] = measure_time(gen_mtc, trials=3)

    return results


def logit_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict:
    """Compare two next-token logit distributions."""
    probs_a = F.softmax(logits_a, dim=-1)
    probs_b = F.softmax(logits_b, dim=-1)

    top1_match = probs_a.argmax(-1) == probs_b.argmax(-1)

    jensen_shannon = 0.5 * (
        F.kl_div(F.log_softmax(logits_a, dim=-1), probs_b, reduction="batchmean")
        + F.kl_div(F.log_softmax(logits_b, dim=-1), probs_a, reduction="batchmean")
    )

    cosine = F.cosine_similarity(logits_a, logits_b, dim=-1).item()

    return {
        "top1_match": top1_match.item(),
        "js_divergence": jensen_shannon.item(),
        "cosine_sim": cosine,
    }
