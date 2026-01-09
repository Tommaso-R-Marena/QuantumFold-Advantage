#!/usr/bin/env python3
"""Run CASP benchmark on trained model.

Usage:
    python scripts/run_casp_benchmark.py \
        --model-path outputs/quantum_run/final_model.pt \
        --casp-version 14 \
        --output results/casp14_benchmark.json
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.advanced_model import AdvancedProteinFoldingModel
from src.protein_embeddings import ESM2Embedder
from src.casp_benchmark import CASPBenchmark


def main():
    parser = argparse.ArgumentParser(description='CASP Benchmarking')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--casp-version', type=int, default=14,
                       choices=[14, 15, 16],
                       help='CASP version to benchmark')
    parser.add_argument('--esm-model', type=str, default='esm2_t33_650M_UR50D',
                       help='ESM-2 model variant')
    parser.add_argument('--output', type=str, default='casp_results.json',
                       help='Output file for results')
    parser.add_argument('--max-targets', type=int, default=None,
                       help='Maximum targets to evaluate (for testing)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("CASP Benchmark Script")
    print(f"{'='*80}\n")
    print(f"Model: {args.model_path}")
    print(f"CASP Version: {args.casp_version}")
    print(f"Device: {args.device}\n")
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load embedder
    print("Loading ESM-2 embedder...")
    embedder = ESM2Embedder(model_name=args.esm_model)
    embedder.to(device)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = AdvancedProteinFoldingModel(
        input_dim=embedder.embed_dim,
        c_s=384,
        c_z=128,
        use_quantum=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully\n")
    
    # Create benchmark
    benchmark = CASPBenchmark(
        model=model,
        embedder=embedder,
        device=device
    )
    
    # Run benchmark
    results = benchmark.benchmark_casp(
        casp_version=args.casp_version,
        max_targets=args.max_targets
    )
    
    # Save results
    if results:
        benchmark.save_results(results, args.output)
        print(f"\nBenchmark complete! Results saved to {args.output}")
    else:
        print("\nNo results generated. Check dataset download.")


if __name__ == '__main__':
    main()
