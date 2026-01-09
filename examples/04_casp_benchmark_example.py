#!/usr/bin/env python3
"""Example: CASP Benchmark Workflow

This script demonstrates the complete workflow for benchmarking your model
on CASP14, CASP15, and CASP16 datasets.

Steps:
1. Download CASP datasets
2. Train quantum and classical models
3. Evaluate on CASP targets
4. Compare results with statistical validation
5. Generate publication-ready plots
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.advanced_model import AdvancedProteinFoldingModel
from src.protein_embeddings import ESM2Embedder
from src.casp_benchmark import CASPDataset, CASPBenchmark
from src.statistical_validation import ComprehensiveBenchmark


def step1_download_datasets():
    """Step 1: Download CASP datasets."""
    print("\n" + "="*80)
    print("STEP 1: Download CASP Datasets")
    print("="*80 + "\n")
    
    for version in [14, 15, 16]:
        print(f"\nDownloading CASP{version}...")
        dataset = CASPDataset(casp_version=version, download=True)
        print(f"  Loaded {len(dataset)} structures")
    
    print("\n✓ Datasets downloaded successfully!")


def step2_quick_test():
    """Step 2: Quick test on a few targets."""
    print("\n" + "="*80)
    print("STEP 2: Quick Test on CASP14 Targets")
    print("="*80 + "\n")
    
    # Load dataset
    dataset = CASPDataset(casp_version=14)
    
    if len(dataset) == 0:
        print("No structures loaded. Please run download first.")
        print("Run: python scripts/download_casp_data.py")
        return
    
    # Initialize embedder
    print("Loading ESM-2 embedder...")
    embedder = ESM2Embedder(model_name='esm2_t6_8M_UR50D')  # Smaller model for demo
    
    # Initialize model
    print("Creating model...")
    model = AdvancedProteinFoldingModel(
        input_dim=320,  # esm2_t6_8M embed dim
        c_s=256,
        c_z=64,
        use_quantum=True,
        num_qubits=4,
        num_quantum_layers=2
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    embedder.to(device)
    
    print(f"Using device: {device}")
    
    # Create benchmark
    benchmark = CASPBenchmark(
        model=model,
        embedder=embedder,
        device=device
    )
    
    # Run on first 5 targets
    print("\nTesting on 5 targets...")
    results = benchmark.benchmark_casp(casp_version=14, max_targets=5)
    
    if results and 'summary' in results:
        print("\n✓ Quick test complete!")
        print(f"  TM-score: {results['summary']['tm_score_mean']:.4f}")
        print(f"  RMSD: {results['summary']['rmsd_mean']:.2f} Å")
    else:
        print("\n⚠ Test completed but no results. Check dataset.")


def step3_full_benchmark():
    """Step 3: Full benchmark (requires trained model)."""
    print("\n" + "="*80)
    print("STEP 3: Full CASP14 Benchmark")
    print("="*80 + "\n")
    
    print("This step requires a trained model.")
    print("\nTo run full benchmark:")
    print("  1. Train your model: python train_advanced.py --use-quantum ...")
    print("  2. Run benchmark: python scripts/run_casp_benchmark.py \\")
    print("       --model-path outputs/quantum_run/final_model.pt \\")
    print("       --casp-version 14 \\")
    print("       --output results/casp14_quantum.json")
    print("  3. Train classical baseline: python train_advanced.py ...")
    print("  4. Benchmark baseline: python scripts/run_casp_benchmark.py \\")
    print("       --model-path outputs/classical_run/final_model.pt \\")
    print("       --casp-version 14 \\")
    print("       --output results/casp14_classical.json")


def step4_statistical_comparison():
    """Step 4: Statistical comparison (mock data for demo)."""
    print("\n" + "="*80)
    print("STEP 4: Statistical Validation")
    print("="*80 + "\n")
    
    # Generate mock data for demonstration
    np.random.seed(42)
    n_targets = 30
    
    # Simulate quantum model results (better)
    quantum_tm_scores = np.random.beta(8, 2, n_targets) * 0.3 + 0.65  # Mean ~0.8
    
    # Simulate classical baseline (slightly worse)
    classical_tm_scores = np.random.beta(8, 2, n_targets) * 0.3 + 0.55  # Mean ~0.7
    
    print("Running statistical validation...")
    print("(Using mock data for demonstration)\n")
    
    # Create benchmark
    stat_benchmark = ComprehensiveBenchmark(output_dir='results/validation')
    
    # Compare methods
    results = stat_benchmark.compare_methods(
        quantum_scores=quantum_tm_scores,
        classical_scores=classical_tm_scores,
        metric_name='TM-score',
        higher_is_better=True
    )
    
    # Print results
    print("\nStatistical Test Results:")
    print(f"  Quantum mean: {np.mean(quantum_tm_scores):.4f} ± {np.std(quantum_tm_scores):.4f}")
    print(f"  Classical mean: {np.mean(classical_tm_scores):.4f} ± {np.std(classical_tm_scores):.4f}")
    print(f"  p-value (Wilcoxon): {results['wilcoxon_p']:.6f}")
    print(f"  Effect size (Cohen's d): {results['cohens_d']:.4f}")
    
    if results['wilcoxon_p'] < 0.05:
        print("  ✓ Statistically significant difference (p < 0.05)")
    else:
        print("  ✗ Not statistically significant (p >= 0.05)")
    
    # Generate comparison plot
    print("\nGenerating comparison plots...")
    stat_benchmark.plot_comparison(
        quantum_tm_scores,
        classical_tm_scores,
        labels=['Quantum', 'Classical']
    )
    
    print("\n✓ Statistical validation complete!")
    print("  Results saved to: results/validation/")


def step5_visualize_results():
    """Step 5: Create publication-ready visualizations."""
    print("\n" + "="*80)
    print("STEP 5: Publication-Ready Visualizations")
    print("="*80 + "\n")
    
    # Mock data
    np.random.seed(42)
    casp_versions = ['CASP14', 'CASP15', 'CASP16']
    
    quantum_scores = {
        'CASP14': np.random.beta(8, 2, 30) * 0.3 + 0.65,
        'CASP15': np.random.beta(8, 2, 25) * 0.3 + 0.65,
        'CASP16': np.random.beta(8, 2, 20) * 0.3 + 0.65,
    }
    
    classical_scores = {
        'CASP14': np.random.beta(8, 2, 30) * 0.3 + 0.55,
        'CASP15': np.random.beta(8, 2, 25) * 0.3 + 0.55,
        'CASP16': np.random.beta(8, 2, 20) * 0.3 + 0.55,
    }
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Bar plot with error bars
    means_q = [np.mean(quantum_scores[v]) for v in casp_versions]
    stds_q = [np.std(quantum_scores[v]) for v in casp_versions]
    means_c = [np.mean(classical_scores[v]) for v in casp_versions]
    stds_c = [np.std(classical_scores[v]) for v in casp_versions]
    
    x = np.arange(len(casp_versions))
    width = 0.35
    
    axes[0].bar(x - width/2, means_q, width, yerr=stds_q, 
                label='Quantum', capsize=5, color='#2E86AB')
    axes[0].bar(x + width/2, means_c, width, yerr=stds_c,
                label='Classical', capsize=5, color='#A23B72')
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_ylabel('TM-score', fontsize=12)
    axes[0].set_title('Performance Across CASP Versions', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(casp_versions)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0.5, 1.0)
    
    # Plot 2: Violin plot for CASP14
    data_for_violin = [
        quantum_scores['CASP14'],
        classical_scores['CASP14']
    ]
    parts = axes[1].violinplot(data_for_violin, positions=[1, 2],
                               showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#2E86AB')
        pc.set_alpha(0.7)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['Quantum', 'Classical'])
    axes[1].set_ylabel('TM-score', fontsize=12)
    axes[1].set_title('CASP14 Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Box plot comparison
    all_quantum = np.concatenate([quantum_scores[v] for v in casp_versions])
    all_classical = np.concatenate([classical_scores[v] for v in casp_versions])
    
    bp = axes[2].boxplot([all_quantum, all_classical],
                         labels=['Quantum\n(All CASP)', 'Classical\n(All CASP)'],
                         patch_artist=True, notch=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')
    axes[2].set_ylabel('TM-score', fontsize=12)
    axes[2].set_title('Overall Comparison', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/casp_benchmark_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: results/casp_benchmark_summary.png")
    plt.close()
    
    print("\n✓ All visualizations complete!")


def main():
    """Run complete CASP benchmark workflow."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  CASP Benchmark Workflow - QuantumFold-Advantage".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Run all steps
    step1_download_datasets()
    step2_quick_test()
    step3_full_benchmark()
    step4_statistical_comparison()
    step5_visualize_results()
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Train your model on a protein dataset")
    print("  2. Run full benchmarks using scripts/run_casp_benchmark.py")
    print("  3. Compare quantum vs classical with statistical validation")
    print("  4. Write your research paper with the results!")
    print("\nFor questions or issues:")
    print("  GitHub: https://github.com/Tommaso-R-Marena/QuantumFold-Advantage")
    print("\n")


if __name__ == '__main__':
    main()
