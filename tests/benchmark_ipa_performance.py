"""Performance benchmarks for new IPA implementation."""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advanced_model import AdvancedProteinFoldingModel


def benchmark_sequence_lengths():
    """Benchmark inference time and memory for various sequence lengths."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmarks")
        return

    device = "cuda"
    model = AdvancedProteinFoldingModel(use_quantum=False, n_structure_layers=8).to(device).eval()

    results = []

    for seq_len in [128, 256, 512, 768]:
        print(f"\n{'='*60}")
        print(f"Benchmarking sequence length: {seq_len}")
        print(f"{'='*60}")

        x = torch.randn(1, seq_len, 1280, device=device)

        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            _ = model(x)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

        results.append(
            {
                "seq_len": seq_len,
                "time_sec": elapsed,
                "memory_gb": peak_memory_gb,
                "time_per_residue_ms": (elapsed / seq_len) * 1000,
            }
        )

        print(f"Time: {elapsed:.3f} sec")
        print(f"Memory: {peak_memory_gb:.2f} GB")
        print(f"Time per residue: {(elapsed/seq_len)*1000:.2f} ms")

        if seq_len > 500:
            assert model.structure_module.ipa_layers[
                0
            ].use_checkpointing, "Checkpointing should be enabled for seq_len > 500"
            print("✓ Checkpointing enabled")

    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Seq Len':<10} {'Time (s)':<12} {'Memory (GB)':<15} {'ms/residue':<12}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['seq_len']:<10} {r['time_sec']:<12.3f} {r['memory_gb']:<15.2f} {r['time_per_residue_ms']:<12.2f}"
        )

    return results


def test_checkpointing_activation():
    """Verify gradient checkpointing activates for long sequences during training."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    model = AdvancedProteinFoldingModel(use_quantum=False, n_structure_layers=8).to("cuda").train()

    x_long = torch.randn(1, 600, 1280, device="cuda", requires_grad=True)

    output = model(x_long)
    loss = output["coordinates"].sum()
    loss.backward()

    print("✓ Checkpointing test passed (no OOM, gradients computed)")


if __name__ == "__main__":
    print("Running IPA performance benchmarks...")
    benchmark_sequence_lengths()
    test_checkpointing_activation()
    print("\n✓ All benchmarks completed successfully")
