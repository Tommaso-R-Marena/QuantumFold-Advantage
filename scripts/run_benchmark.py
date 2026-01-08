#!/usr/bin/env python3
"""Run comprehensive benchmarking against AlphaFold-3 and baselines.

Usage:
    python scripts/run_benchmark.py --test_set data/casp15_test.json \
                                     --output_dir results/benchmarks
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import BenchmarkComparison, ProteinStructureEvaluator
from src.model import QuantumFoldModel
from src.data import load_protein_structure


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark QuantumFold against baselines')
    parser.add_argument('--test_set', type=str, required=True,
                        help='Path to test set JSON file')
    parser.add_argument('--model_checkpoint', type=str,
                        default='checkpoints/quantumfold_best.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/benchmarks',
                        help='Directory to save benchmark results')
    parser.add_argument('--compare_alphafold', action='store_true',
                        help='Include AlphaFold-3 comparison')
    parser.add_argument('--alphafold_dir', type=str,
                        help='Directory containing AlphaFold-3 predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    return parser.parse_args()


def load_test_set(test_set_path: str) -> list:
    """Load test set configuration."""
    with open(test_set_path, 'r') as f:
        return json.load(f)


def run_quantumfold_inference(model, protein_data, device):
    """Run QuantumFold inference on a protein."""
    # Prepare input
    sequence = protein_data['sequence']
    
    # Convert sequence to input features (simplified)
    # In practice, this would use MSA, templates, etc.
    input_features = prepare_features(sequence)
    input_features = {k: v.to(device) for k, v in input_features.items()}
    
    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(input_features)
    
    # Extract coordinates
    coords = predictions['coordinates'].cpu().numpy()
    return coords


def prepare_features(sequence: str) -> dict:
    """Prepare model input features from sequence.
    
    This is a simplified version. Full implementation would include:
    - Multiple sequence alignment (MSA)
    - Template structures
    - Residue embeddings
    - Distance maps
    """
    seq_len = len(sequence)
    
    # Amino acid encoding (one-hot)
    aa_types = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(aa_types)}
    
    encoding = torch.zeros(seq_len, len(aa_types))
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            encoding[i, aa_to_idx[aa]] = 1.0
    
    return {
        'sequence_encoding': encoding.unsqueeze(0),  # Add batch dimension
        'sequence_length': torch.tensor([seq_len])
    }


def main():
    args = parse_args()
    
    print("="*80)
    print("QuantumFold Benchmarking Suite")
    print("="*80)
    print(f"Test set: {args.test_set}")
    print(f"Model: {args.model_checkpoint}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test set
    print("Loading test set...")
    test_proteins = load_test_set(args.test_set)
    print(f"Loaded {len(test_proteins)} proteins for evaluation\n")
    
    # Load model
    print("Loading QuantumFold model...")
    model = QuantumFoldModel(
        n_layers=6,
        embed_dim=128,
        n_heads=8,
        use_quantum=True,
        n_qubits=4
    ).to(args.device)
    
    if Path(args.model_checkpoint).exists():
        checkpoint = torch.load(args.model_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}\n")
    else:
        print(f"Warning: Checkpoint not found at {args.model_checkpoint}")
        print("Using randomly initialized model\n")
    
    # Initialize benchmark comparison
    benchmark = BenchmarkComparison(output_dir=str(output_dir))
    
    # Run evaluation on each protein
    print("Running predictions and evaluations...\n")
    for i, protein_data in enumerate(test_proteins):
        protein_id = protein_data['id']
        print(f"[{i+1}/{len(test_proteins)}] Evaluating {protein_id}...")
        
        try:
            # Load ground truth structure
            coords_true = load_protein_structure(protein_data['pdb_path'])
            
            # Run QuantumFold prediction
            coords_quantumfold = run_quantumfold_inference(
                model, protein_data, args.device
            )
            
            # Load AlphaFold-3 predictions if available
            coords_alphafold = None
            if args.compare_alphafold and args.alphafold_dir:
                alphafold_path = Path(args.alphafold_dir) / f"{protein_id}_alphafold.pdb"
                if alphafold_path.exists():
                    coords_alphafold = load_protein_structure(str(alphafold_path))
            
            # Compare predictions
            comparison = benchmark.compare_predictions(
                protein_id=protein_id,
                coords_true=coords_true,
                coords_quantumfold=coords_quantumfold,
                coords_alphafold=coords_alphafold,
                sequence_length=len(protein_data['sequence'])
            )
            
            # Print results for this protein
            for method, metrics in comparison.items():
                print(f"  {method}: TM-score={metrics.tm_score:.3f}, "
                      f"RMSD={metrics.rmsd:.2f}Å, GDT_TS={metrics.gdt_ts:.1f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        print()
    
    # Save results
    print("\nSaving benchmark results...")
    benchmark.save_results('benchmark_results.json')
    
    # Print summary
    benchmark.print_summary()
    
    print("\nBenchmarking complete!")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
