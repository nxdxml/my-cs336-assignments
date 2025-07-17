import torch
import time
import numpy as np
from tqdm import tqdm
from cs336_basics.Transformer import TransformerLM

import torch.cuda.nvtx as nvtx

# -------------------------
# Benchmarking Script
# -------------------------

def benchmark_model(
    d_model=128,
    num_layers=4,
    num_heads=4,
    d_ff=None,
    context_length=64,
    vocab_size=10000,
    batch_size=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    warmup_steps=5,
    measure_steps=10,
    do_backward=False,
    rope_theta=10000,
):
    print("\n--- Benchmarking Configuration ---")
    print(f"Device: {device}")
    print(f"Model: d_model={d_model}, layers={num_layers}, heads={num_heads}, d_ff={d_ff or 4*d_model}")
    print(f"Context length: {context_length}, Batch size: {batch_size}")
    print(f"Backward Pass: {do_backward}")

    # Set FF dim
    d_ff = d_ff or 4 * d_model

    # Create model and move to device
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device,
        rope_theta=rope_theta,
    ).to(device)

    # Optimizer for backward pass
    optimizer = torch.optim.AdamW(model.parameters()) if do_backward else None

    # Create random input and targets
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
    targets = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)

    # Warm-up steps (important for accurate timing)
    for _ in range(warmup_steps):
        logits = model(x)
        if do_backward:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets_reshaped = targets.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits, targets_reshaped)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize() if device == "cuda" else None

    # Actual timing
    times = []
    for _ in tqdm(range(measure_steps), desc="Benchmarking"):
        start = time.time()
        # nvtx标记
        with nvtx.range("forward pass"):
            logits = model(x)
        if do_backward:
            with nvtx.range("backward pass"):
                B, T, V = logits.shape
                logits = logits.view(B * T, V)
                targets_reshaped = targets.view(B * T)
                loss = torch.nn.functional.cross_entropy(logits, targets_reshaped)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.time()
        times.append(end - start)

    # Compute stats
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\nAvg time per step ({'forward+backward' if do_backward else 'forward'}): {avg_time:.6f}s (std: {std_time:.6f}s)")


# -------------------------
# Run benchmarks for different model sizes
# -------------------------

if __name__ == "__main__":
    configs = {
        "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
        "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
        # "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
        # "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
        # "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),  # Optional, very large
    }

    for name, cfg in configs.items():
        print(f"\n===== Benchmarking: {name} model =====")
        benchmark_model(
            **cfg,
            context_length=64,
            batch_size=4,
            warmup_steps=5,
            measure_steps=10,
            do_backward=False,
            rope_theta=10000,
        )
        benchmark_model(
            **cfg,
            context_length=64,
            batch_size=4,
            warmup_steps=5,
            measure_steps=10,
            do_backward=True,
            rope_theta=10000,
        )
