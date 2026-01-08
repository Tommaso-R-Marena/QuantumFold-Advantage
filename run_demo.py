#!/usr/bin/env python3
"""
QuantumFold-Advantage Main Entrypoint

Runs the full pipeline: fetch data → train model → evaluate → visualize.
Outputs saved to outputs/ directory.

Usage:
    python run_demo.py
    python run_demo.py --seed 42 --mode cpu
    python run_demo.py --quantum --epochs 20
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import run_full_pipeline
from src.utils import setup_logging, set_seed

# Default configuration
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QuantumFold-Advantage: Hybrid Quantum-Classical Protein Folding"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Device mode (auto-detects by default)",
    )
    parser.add_argument(
        "--quantum",
        action="store_true",
        help="Enable hybrid quantum-classical model (requires PennyLane)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("QuantumFold-Advantage: Hybrid Protein Folding Pipeline")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Device Mode: {args.mode}")
    logger.info(f"  Quantum Mode: {args.quantum}")
    logger.info(f"  Output Directory: {args.output_dir}")
    logger.info("=" * 60)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run full pipeline
    try:
        results = run_full_pipeline(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device_mode=args.mode,
            use_quantum=args.quantum,
            output_dir=str(output_dir),
        )
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"  - Training metrics: {output_dir / 'train_metrics.json'}")
        logger.info(f"  - Predictions: {output_dir / 'predictions.pdb'}")
        logger.info(f"  - Visualization: {output_dir / 'structure_viz.html'}")
        logger.info("=" * 60)
        
        # Print summary statistics
        if "test_rmsd" in results:
            logger.info(f"Test RMSD: {results['test_rmsd']:.4f} Å")
        if "test_tm_score" in results:
            logger.info(f"Test TM-Score: {results['test_tm_score']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())