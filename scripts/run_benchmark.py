#!/usr/bin/env python
"""Run comprehensive benchmarking against AlphaFold-3 and baselines.

Usage:
    python scripts/run_benchmark.py --config configs/benchmark_config.yaml
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import QuantumFoldModel
from src.benchmarks import ProteinBenchmark
from src.data import ProteinDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_test_data(data_dir: str, split: str = 'test'):
    """Load test dataset."""
    dataset = ProteinDataset(
        data_dir=data_dir,
        split=split,
        max_seq_length=512
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark QuantumFold against AlphaFold-3'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/benchmark_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmarks',
        help='Directory for benchmark outputs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Initialize model
    print(f"Loading model from {args.model_checkpoint}...")
    model = QuantumFoldModel(**config['model'])
    checkpoint = torch.load(args.model_checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    print("Loading test data...")
    test_dataset = load_test_data(
        data_dir=config['data']['test_dir'],
        split='test'
    )
    
    # Prepare test data in format for benchmarking
    test_data = []
    for i in range(len(test_dataset)):
        features, coords, metadata = test_dataset[i]
        protein_id = metadata.get('protein_id', f'protein_{i}')
        test_data.append((features, coords.numpy(), protein_id))
    
    # Initialize benchmark suite
    print("Initializing benchmark suite...")
    benchmark = ProteinBenchmark(output_dir=args.output_dir)
    
    # Run evaluation
    print(f"\nEvaluating {len(test_data)} proteins...")
    results = benchmark.evaluate_model(
        model=model,
        test_data=test_data,
        model_name="QuantumFold",
        device=args.device
    )
    
    # Print results
    print("\n" + benchmark.generate_report())
    
    # Save results
    benchmark.save_results(filename="quantumfold_benchmark.json")
    
    # Compare with AlphaFold-3 if predictions available
    if 'alphafold3_predictions' in config:
        print("\nComparing with AlphaFold-3...")
        
        # Load AF3 predictions (placeholder - implement based on format)
        af3_predictions = {}  # Load from config path
        ground_truth = {}
        
        comparison = benchmark.compare_with_alphafold3(
            alphafold_predictions=af3_predictions,
            ground_truth=ground_truth
        )
        
        print("\nComparison with AlphaFold-3:")
        print(f"  QuantumFold mean TM-score: {comparison['quantumfold_mean_tm']:.4f}")
        print(f"  AlphaFold-3 mean TM-score: {comparison['alphafold3_mean_tm']:.4f}")
        print(f"  Improvement: {comparison['tm_improvement']:.4f}")
    
    print(f"\nBenchmark complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
