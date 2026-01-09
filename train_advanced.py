#!/usr/bin/env python3
"""Advanced training script for QuantumFold-Advantage.

Integrates all state-of-the-art components:
- ESM-2 pre-trained embeddings
- Invariant Point Attention (IPA)
- Frame Aligned Point Error (FAPE) loss
- Mixed precision training
- Statistical validation
- Comprehensive logging

Usage:
    python train_advanced.py --config configs/advanced_config.yaml
    python train_advanced.py --model quantum --epochs 100 --batch-size 32

Author: Tommaso R. Marena
Institution: The Catholic University of America
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Optional
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Import advanced modules
from src.advanced_model import AdvancedProteinFoldingModel
from src.advanced_training import AdvancedTrainer, TrainingConfig
from src.protein_embeddings import ESM2Embedder, CombinedProteinEmbedding
from src.statistical_validation import ComprehensiveBenchmark
from src.benchmarks import ProteinStructureEvaluator


class ProteinDataset(Dataset):
    """Simple protein dataset for demonstration."""
    
    def __init__(self, sequences: list, coordinates: list):
        self.sequences = sequences
        self.coordinates = coordinates
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'coordinates': torch.tensor(self.coordinates[idx], dtype=torch.float32)
        }


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = log_dir / 'training.log'
    
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('QuantumFold')
    logger.info('Logging initialized')
    logger.info(f'Log file: {log_file}')
    
    return logger


def create_synthetic_data(n_samples: int = 100, seq_len: int = 50) -> tuple:
    """Create synthetic protein data for testing.
    
    Args:
        n_samples: Number of protein samples
        seq_len: Sequence length
    
    Returns:
        (sequences, coordinates) tuple
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    sequences = []
    coordinates = []
    
    for _ in range(n_samples):
        # Random sequence
        seq = ''.join(np.random.choice(list(amino_acids), size=seq_len))
        sequences.append(seq)
        
        # Alpha helix coordinates
        t = np.linspace(0, 4*np.pi, seq_len)
        coords = np.zeros((seq_len, 3))
        coords[:, 0] = 2.3 * np.cos(t)
        coords[:, 1] = 2.3 * np.sin(t)
        coords[:, 2] = 1.5 * t
        coords += np.random.randn(seq_len, 3) * 0.5  # Add noise
        
        coordinates.append(coords)
    
    return sequences, coordinates


def collate_fn_with_embeddings(batch: list, embedder: ESM2Embedder) -> Dict:
    """Custom collate function that generates embeddings.
    
    Args:
        batch: List of samples
        embedder: ESM-2 embedder
    
    Returns:
        Batch dictionary
    """
    sequences = [item['sequence'] for item in batch]
    coordinates = torch.stack([item['coordinates'] for item in batch])
    
    # Generate embeddings
    with torch.no_grad():
        embeddings_dict = embedder(sequences)
        embeddings = embeddings_dict['embeddings']
    
    return {
        'sequence': embeddings,
        'coordinates': coordinates,
        'mask': torch.ones(len(batch), len(sequences[0]), dtype=torch.bool)
    }


def train(args: argparse.Namespace):
    """Main training function."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir / 'logs', args.verbose)
    logger.info(f"QuantumFold-Advantage Training")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create synthetic data (replace with real data loading)
    logger.info("Creating synthetic training data...")
    train_sequences, train_coordinates = create_synthetic_data(
        n_samples=args.train_samples,
        seq_len=args.seq_len
    )
    val_sequences, val_coordinates = create_synthetic_data(
        n_samples=args.val_samples,
        seq_len=args.seq_len
    )
    
    # Initialize ESM-2 embedder
    logger.info("Loading ESM-2 embedder...")
    embedder = ESM2Embedder(
        model_name=args.esm_model,
        freeze=True
    ).to(device)
    
    # Create datasets
    train_dataset = ProteinDataset(train_sequences, train_coordinates)
    val_dataset = ProteinDataset(val_sequences, val_coordinates)
    
    # Create dataloaders with embedding collation
    from functools import partial
    collate_with_emb = partial(collate_fn_with_embeddings, embedder=embedder)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_with_emb,
        num_workers=0  # ESM-2 doesn't work well with multiprocessing
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_with_emb,
        num_workers=0
    )
    
    # Initialize model
    logger.info(f"Initializing model (quantum={args.use_quantum})...")
    model = AdvancedProteinFoldingModel(
        input_dim=embedder.embed_dim,
        c_s=args.hidden_dim,
        c_z=args.pair_dim,
        n_structure_layers=args.n_structure_layers,
        use_quantum=args.use_quantum
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Training configuration
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        checkpoint_dir=str(output_dir / 'checkpoints'),
        warmup_epochs=args.warmup_epochs,
        gradient_clip_norm=args.grad_clip
    )
    
    # Initialize trainer
    trainer = AdvancedTrainer(
        model=model,
        config=config,
        device=device,
        logger=logger
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'history': history
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Statistical validation (if enabled)
    if args.run_validation:
        logger.info("Running statistical validation...")
        run_statistical_validation(model, val_loader, output_dir, device, logger)
    
    logger.info("Training complete!")
    return history


def run_statistical_validation(
    model: nn.Module,
    val_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    logger: logging.Logger
):
    """Run comprehensive statistical validation."""
    
    model.eval()
    evaluator = ProteinStructureEvaluator()
    benchmark = ComprehensiveBenchmark(output_dir=str(output_dir / 'validation'))
    
    # Collect predictions
    all_predictions = []
    all_ground_truth = []
    all_tm_scores = []
    all_rmsd_scores = []
    
    logger.info("Collecting predictions for validation...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs = batch['sequence'].to(device)
            coords_true = batch['coordinates'].to(device)
            
            # Predict
            outputs = model(inputs)
            coords_pred = outputs['coordinates']
            
            # Move to CPU and numpy
            coords_pred_np = coords_pred.cpu().numpy()
            coords_true_np = coords_true.cpu().numpy()
            
            # Compute metrics for each sample
            for i in range(len(coords_pred_np)):
                pred = coords_pred_np[i]
                true = coords_true_np[i]
                
                # Calculate metrics
                rmsd = evaluator.calculate_rmsd(pred, true)
                tm_score = evaluator.calculate_tm_score(pred, true)
                
                all_predictions.append(pred)
                all_ground_truth.append(true)
                all_tm_scores.append(tm_score)
                all_rmsd_scores.append(rmsd)
    
    # Convert to arrays
    tm_scores = np.array(all_tm_scores)
    rmsd_scores = np.array(all_rmsd_scores)
    
    # Summary statistics
    logger.info("\nValidation Results:")
    logger.info(f"TM-score: {tm_scores.mean():.4f} ± {tm_scores.std():.4f}")
    logger.info(f"RMSD: {rmsd_scores.mean():.4f} ± {rmsd_scores.std():.4f}")
    
    # Create comparison (simulate classical baseline for demonstration)
    classical_tm_scores = tm_scores - np.random.randn(len(tm_scores)) * 0.05
    
    # Statistical comparison
    results = benchmark.compare_methods(
        quantum_scores=tm_scores,
        classical_scores=classical_tm_scores,
        metric_name='TM-score',
        higher_is_better=True
    )
    
    # Plot comparison
    benchmark.plot_comparison(
        quantum_scores=tm_scores,
        classical_scores=classical_tm_scores,
        metric_name='TM-score'
    )
    
    # Save results
    benchmark.save_results()
    benchmark.generate_report()
    
    logger.info(f"Statistical validation complete. Results saved to {output_dir / 'validation'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Advanced training for QuantumFold-Advantage',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-samples', type=int, default=100,
                       help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=20,
                       help='Number of validation samples')
    parser.add_argument('--seq-len', type=int, default=50,
                       help='Protein sequence length')
    
    # Model arguments
    parser.add_argument('--esm-model', type=str, default='esm2_t33_650M_UR50D',
                       choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D',
                               'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'],
                       help='ESM-2 model variant')
    parser.add_argument('--hidden-dim', type=int, default=384,
                       help='Hidden dimension')
    parser.add_argument('--pair-dim', type=int, default=128,
                       help='Pair representation dimension')
    parser.add_argument('--n-structure-layers', type=int, default=8,
                       help='Number of structure refinement layers')
    parser.add_argument('--use-quantum', action='store_true',
                       help='Enable quantum enhancement')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # Advanced training features
    parser.add_argument('--use-amp', action='store_true',
                       help='Enable automatic mixed precision')
    parser.add_argument('--use-ema', action='store_true',
                       help='Enable exponential moving average')
    
    # Validation
    parser.add_argument('--run-validation', action='store_true',
                       help='Run statistical validation after training')
    
    # System
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*80)
    print("QuantumFold-Advantage Training Configuration")
    print("="*80)
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg:20s}: {value}")
    print("="*80)
    
    # Train
    try:
        history = train(args)
        print("\n" + "="*80)
        print("Training completed successfully!")
        print("="*80)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
