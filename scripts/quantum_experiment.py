#!/usr/bin/env python3
"""Orchestrate paired quantum vs. classical experiments."""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, detect_device, setup_logging
from src.pipeline import run_pipeline

logger = logging.getLogger(__name__)

def run_experiment(
    num_runs=5,
    num_samples=100,
    epochs=10,
    output_dir='outputs/quantum_experiment'
):
    """
    Run paired quantum vs. classical experiments.
    
    Args:
        num_runs: Number of independent runs
        num_samples: Samples per run
        epochs: Training epochs
        output_dir: Output directory
    
    Returns:
        dict: Experiment results
    """
    device = detect_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'num_runs': num_runs,
        'num_samples': num_samples,
        'epochs': epochs,
        'device': device,
        'classical': [],
        'hybrid': []
    }
    
    for run_idx in range(num_runs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {run_idx + 1}/{num_runs}")
        logger.info(f"{'='*60}")
        
        seed = 42 + run_idx  # Different seed per run
        
        # Classical baseline
        logger.info(f"\nRunning CLASSICAL baseline (run {run_idx + 1})...")
        set_seed(seed)
        classical_result = run_pipeline(
            mode='classical',
            device=device,
            num_samples=num_samples,
            epochs=epochs,
            seed=seed,
            output_dir=str(output_path / f'classical_run{run_idx}'),
            visualize=False
        )
        results['classical'].append({
            'run': run_idx,
            'seed': seed,
            'final_val_loss': classical_result['final_val_loss'],
            'final_train_loss': classical_result['final_train_loss']
        })
        
        # Hybrid quantum
        logger.info(f"\nRunning HYBRID quantum (run {run_idx + 1})...")
        set_seed(seed)  # Same seed for fair comparison
        hybrid_result = run_pipeline(
            mode='hybrid',
            device=device,
            num_samples=num_samples,
            epochs=epochs,
            seed=seed,
            output_dir=str(output_path / f'hybrid_run{run_idx}'),
            visualize=False
        )
        results['hybrid'].append({
            'run': run_idx,
            'seed': seed,
            'final_val_loss': hybrid_result['final_val_loss'],
            'final_train_loss': hybrid_result['final_train_loss']
        })
    
    # Save results
    results_file = output_path / 'experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nExperiment results saved to {results_file}")
    return results

if __name__ == '__main__':
    setup_logging()
    
    results = run_experiment(
        num_runs=5,
        num_samples=100,
        epochs=10
    )
    
    # Quick summary
    import numpy as np
    classical_losses = [r['final_val_loss'] for r in results['classical']]
    hybrid_losses = [r['final_val_loss'] for r in results['hybrid']]
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Classical mean val loss: {np.mean(classical_losses):.4f} ± {np.std(classical_losses):.4f}")
    print(f"Hybrid mean val loss: {np.mean(hybrid_losses):.4f} ± {np.std(hybrid_losses):.4f}")
    print("\nRun statistical_evaluation.py for significance testing.")
    print("="*60)