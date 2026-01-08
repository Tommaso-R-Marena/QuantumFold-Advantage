#!/usr/bin/env python3
"""Create publication-quality comparison figures.

Usage:
    python scripts/visualize_comparison.py \
        --predicted predictions/protein.pdb \
        --ground_truth data/protein_true.pdb \
        --output figures/comparison.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_protein_structure
from src.benchmarks import ProteinStructureEvaluator


def create_comparison_figure(
    coords_pred: np.ndarray,
    coords_true: np.ndarray,
    sequence: str,
    output_path: str,
    title: str = "Structure Prediction Comparison"
):
    """Create comprehensive comparison figure.
    
    Args:
        coords_pred: Predicted coordinates
        coords_true: Ground truth coordinates
        sequence: Amino acid sequence
        output_path: Output file path
        title: Figure title
    """
    # Calculate metrics
    evaluator = ProteinStructureEvaluator()
    metrics = evaluator.evaluate_structure(coords_pred, coords_true, len(sequence))
    
    # Calculate distances
    dist_pred = np.sqrt(np.sum((coords_pred[:, None, :] - coords_pred[None, :, :]) ** 2, axis=2))
    dist_true = np.sqrt(np.sum((coords_true[:, None, :] - coords_true[None, :, :]) ** 2, axis=2))
    dist_diff = np.abs(dist_pred - dist_true)
    
    # Calculate per-residue RMSD
    per_residue_rmsd = np.sqrt(np.sum((coords_pred - coords_true) ** 2, axis=1))
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 3D overlay (top left, span 2 columns)
    ax1 = fig.add_subplot(gs[0, :2], projection='3d')
    ax1.plot(coords_true[:, 0], coords_true[:, 1], coords_true[:, 2], 
             'b-', linewidth=2, alpha=0.7, label='Ground Truth')
    ax1.plot(coords_pred[:, 0], coords_pred[:, 1], coords_pred[:, 2], 
             'r-', linewidth=2, alpha=0.7, label='Predicted')
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.set_title('3D Structure Overlay')
    ax1.legend()
    
    # 2. Distance map comparison (top right, 2x2 grid)
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(dist_true, cmap='viridis', interpolation='nearest')
    ax2.set_title('True Distance Map')
    ax2.set_xlabel('Residue')
    ax2.set_ylabel('Residue')
    plt.colorbar(im2, ax=ax2, label='Distance (Å)')
    
    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(dist_pred, cmap='viridis', interpolation='nearest')
    ax3.set_title('Predicted Distance Map')
    ax3.set_xlabel('Residue')
    ax3.set_ylabel('Residue')
    plt.colorbar(im3, ax=ax3, label='Distance (Å)')
    
    # 3. Distance difference (middle left)
    ax4 = fig.add_subplot(gs[1, :2])
    im4 = ax4.imshow(dist_diff, cmap='hot', interpolation='nearest', vmin=0, vmax=5)
    ax4.set_title('Distance Difference Map')
    ax4.set_xlabel('Residue')
    ax4.set_ylabel('Residue')
    plt.colorbar(im4, ax=ax4, label='|Δ Distance| (Å)')
    
    # 4. Per-residue RMSD (middle right)
    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.plot(per_residue_rmsd, 'o-', linewidth=2, markersize=4, color='steelblue')
    ax5.axhline(y=np.mean(per_residue_rmsd), color='r', linestyle='--', 
                label=f'Mean: {np.mean(per_residue_rmsd):.2f} Å')
    ax5.set_xlabel('Residue Index')
    ax5.set_ylabel('RMSD (Å)')
    ax5.set_title('Per-Residue RMSD')
    ax5.grid(alpha=0.3)
    ax5.legend()
    
    # 5. Metrics summary (bottom left)
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.axis('off')
    
    metrics_text = f"""
    EVALUATION METRICS
    
    RMSD:           {metrics.rmsd:.2f} Å
    TM-score:       {metrics.tm_score:.3f}
    GDT_TS:         {metrics.gdt_ts:.1f}
    GDT_HA:         {metrics.gdt_ha:.1f}
    lDDT:           {metrics.lddt:.3f}
    Clash Score:    {metrics.clash_score:.2f}
    
    Sequence Length: {len(sequence)} residues
    """
    
    ax6.text(0.1, 0.5, metrics_text, fontsize=14, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Distribution comparison (bottom right)
    ax7 = fig.add_subplot(gs[2, 2:])
    
    # Calculate contact distributions
    contact_threshold = 8.0
    contacts_true = dist_true[np.triu_indices_from(dist_true, k=1)]
    contacts_pred = dist_pred[np.triu_indices_from(dist_pred, k=1)]
    
    ax7.hist(contacts_true, bins=50, alpha=0.5, label='Ground Truth', color='blue')
    ax7.hist(contacts_pred, bins=50, alpha=0.5, label='Predicted', color='red')
    ax7.axvline(contact_threshold, color='green', linestyle='--', 
                label=f'Contact Threshold ({contact_threshold} Å)')
    ax7.set_xlabel('Distance (Å)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Pairwise Distance Distribution')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Create comparison visualization')
    parser.add_argument('--predicted', required=True, help='Predicted structure PDB')
    parser.add_argument('--ground_truth', required=True, help='Ground truth structure PDB')
    parser.add_argument('--output', default='comparison.png', help='Output figure path')
    parser.add_argument('--title', default='Structure Prediction Comparison', help='Figure title')
    
    args = parser.parse_args()
    
    # Load structures
    print(f"Loading predicted structure from {args.predicted}...")
    coords_pred = load_protein_structure(args.predicted)
    
    print(f"Loading ground truth from {args.ground_truth}...")
    coords_true = load_protein_structure(args.ground_truth)
    
    # Get sequence (simplified - would parse from PDB)
    sequence = 'A' * len(coords_true)  # Placeholder
    
    # Create figure
    print("Creating comparison figure...")
    metrics = create_comparison_figure(
        coords_pred,
        coords_true,
        sequence,
        args.output,
        args.title
    )
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in metrics.to_dict().items():
        print(f"{key:15s}: {value:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
