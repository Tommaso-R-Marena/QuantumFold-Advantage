#!/usr/bin/env python3
"""Evaluate trained QuantumFold model on validation set.

Usage:
    python scripts/evaluate_model.py --checkpoint checkpoints/best_model.pt \
                                      --data_dir data/validation
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import QuantumFoldModel
from src.data import ProteinDataset, create_dataloader
from src.benchmarks import ProteinStructureEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate QuantumFold model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing validation data')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--use_quantum', action='store_true',
                        help='Use quantum layers (must match training)')
    return parser.parse_args()


def evaluate_model(model, dataloader, evaluator, device):
    """Evaluate model on dataset."""
    model.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            coords_true = batch['coords'].cpu().numpy()
            
            # Forward pass
            outputs = model(inputs)
            coords_pred = outputs['coordinates'].cpu().numpy()
            
            # Evaluate each sample in batch
            batch_size = coords_pred.shape[0]
            for i in range(batch_size):
                metrics = evaluator.evaluate_structure(
                    coords_pred[i],
                    coords_true[i],
                    sequence_length=batch['sequence_length'][i].item()
                )
                all_metrics.append(metrics.to_dict())
    
    return all_metrics


def compute_statistics(metrics_list):
    """Compute mean and std for all metrics."""
    metrics_names = metrics_list[0].keys()
    stats = {}
    
    for metric_name in metrics_names:
        values = [m[metric_name] for m in metrics_list]
        stats[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    return stats


def main():
    args = parse_args()
    
    print("QuantumFold Model Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {args.device}")
    print(f"Quantum layers: {args.use_quantum}")
    print()
    
    # Load model
    print("Loading model...")
    model = QuantumFoldModel(
        n_layers=6,
        embed_dim=128,
        n_heads=8,
        use_quantum=args.use_quantum,
        n_qubits=4
    ).to(args.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    train_loss = checkpoint.get('loss', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch} (train loss: {train_loss})")
    print()
    
    # Load data
    print("Loading validation dataset...")
    dataset = ProteinDataset(args.data_dir, augment=False)
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    print(f"Dataset size: {len(dataset)} proteins\n")
    
    # Initialize evaluator
    evaluator = ProteinStructureEvaluator()
    
    # Run evaluation
    print("Running evaluation...")
    metrics_list = evaluate_model(model, dataloader, evaluator, args.device)
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(metrics_list)
    
    # Prepare results
    results = {
        'checkpoint': args.checkpoint,
        'epoch': epoch,
        'n_samples': len(metrics_list),
        'statistics': stats,
        'individual_results': metrics_list
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for metric_name, metric_stats in stats.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean:   {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}")
        print(f"  Median: {metric_stats['median']:.4f}")
        print(f"  Range:  [{metric_stats['min']:.4f}, {metric_stats['max']:.4f}]")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
