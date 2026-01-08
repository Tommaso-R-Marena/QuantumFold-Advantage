#!/usr/bin/env python3
"""Benchmarking script for QuantumFold-Advantage."""
import sys
import time
import json
import logging
import psutil
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, detect_device, setup_logging
from src.pipeline import run_pipeline

logger = logging.getLogger(__name__)

def benchmark_run(mode, device, num_samples=50, epochs=5):
    """
    Run benchmark for a specific configuration.
    
    Returns:
        dict: Benchmark results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {mode} mode on {device}")
    logger.info(f"{'='*60}")
    
    # Memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2  # MB
    
    # Time the pipeline
    start_time = time.time()
    
    try:
        results = run_pipeline(
            mode=mode,
            device=device,
            num_samples=num_samples,
            epochs=epochs,
            batch_size=8,
            hidden_dim=64,
            seed=42,
            output_dir=f'outputs/benchmark_{mode}_{device}',
            visualize=False  # Skip visualization for speed
        )
        
        elapsed_time = time.time() - start_time
        
        # Memory after
        mem_after = process.memory_info().rss / 1024**2  # MB
        
        benchmark_results = {
            'mode': mode,
            'device': device,
            'elapsed_time': elapsed_time,
            'memory_used_mb': mem_after - mem_before,
            'final_train_loss': results['final_train_loss'],
            'final_val_loss': results['final_val_loss'],
            'num_samples': num_samples,
            'epochs': epochs,
            'success': True
        }
        
        logger.info(f"Completed in {elapsed_time:.2f}s")
        logger.info(f"Memory used: {benchmark_results['memory_used_mb']:.2f} MB")
        logger.info(f"Final validation loss: {results['final_val_loss']:.4f}")
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {
            'mode': mode,
            'device': device,
            'success': False,
            'error': str(e)
        }

if __name__ == '__main__':
    setup_logging()
    set_seed(42)
    
    # Detect device
    device = detect_device()
    
    # Run benchmarks
    results = []
    
    # Classical baseline
    results.append(benchmark_run('classical', device, num_samples=50, epochs=5))
    
    # Hybrid quantum-classical
    results.append(benchmark_run('hybrid', device, num_samples=50, epochs=5))
    
    # Save results
    output_path = Path('outputs/benchmark_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nBenchmark results saved to {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for r in results:
        if r['success']:
            print(f"\n{r['mode'].upper()} on {r['device'].upper()}:")
            print(f"  Time: {r['elapsed_time']:.2f}s")
            print(f"  Memory: {r['memory_used_mb']:.2f} MB")
            print(f"  Final Val Loss: {r['final_val_loss']:.4f}")
        else:
            print(f"\n{r['mode'].upper()} FAILED: {r.get('error', 'Unknown')}")
    print("\n" + "="*60)