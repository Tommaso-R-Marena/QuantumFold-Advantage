"""Orchestrate training and evaluation pipelines."""
import logging
import json
from pathlib import Path
import torch
from .utils import set_seed
from .data import load_or_generate_data
from .model import ProteinFoldNet
from .train import train_model
from .visualize import visualize_protein_structure

logger = logging.getLogger(__name__)

def run_pipeline(
    mode='classical',
    device='cpu',
    num_samples=50,
    max_len=30,
    epochs=10,
    batch_size=8,
    hidden_dim=64,
    seed=42,
    output_dir='outputs',
    visualize=True,
):
    """
    Run end-to-end protein folding pipeline.
    
    Args:
        mode: 'classical' or 'hybrid'
        device: 'cpu' or 'cuda'
        num_samples: Number of protein samples
        max_len: Maximum sequence length
        epochs: Training epochs
        batch_size: Batch size
        hidden_dim: Hidden dimension
        seed: Random seed
        output_dir: Output directory
        visualize: Whether to generate visualizations
    
    Returns:
        dict: Results including metrics and paths
    """
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running pipeline: mode={mode}, device={device}, samples={num_samples}")
    
    # Load data
    logger.info("Loading/generating data...")
    train_loader, val_loader = load_or_generate_data(
        num_samples=num_samples,
        max_len=max_len,
        batch_size=batch_size,
        device=device
    )
    
    # Build model
    logger.info(f"Building {mode} model...")
    use_quantum = (mode == 'hybrid')
    model = ProteinFoldNet(
        vocab_size=21,  # 20 amino acids + padding
        hidden_dim=hidden_dim,
        output_dim=3,   # x, y, z coordinates
        use_quantum=use_quantum,
        n_qubits=4 if use_quantum else None
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    logger.info("Training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device
    )
    
    # Save results
    results = {
        'mode': mode,
        'device': device,
        'num_samples': num_samples,
        'max_len': max_len,
        'epochs': epochs,
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'seed': seed,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'history': history
    }
    
    results_file = output_path / f'results_{mode}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # Visualization
    if visualize:
        logger.info("Generating visualization...")
        # Get a sample prediction
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            sequences, coords = sample_batch
            pred_coords = model(sequences)
            
            # Take first sample
            sample_seq = sequences[0].cpu().numpy()
            sample_pred = pred_coords[0].cpu().numpy()
            sample_true = coords[0].cpu().numpy()
            
            # Convert sequence indices to amino acids (simplified)
            aa_map = 'ACDEFGHIKLMNPQRSTVWY'
            sequence_str = ''.join([aa_map[min(i, 19)] for i in sample_seq if i > 0])
            
            # Visualize
            pdb_path = output_path / f'structure_{mode}.pdb'
            visualize_protein_structure(
                coords=sample_pred,
                sequence=sequence_str,
                output_path=str(pdb_path)
            )
            results['pdb_path'] = str(pdb_path)
            logger.info(f"PDB structure saved to {pdb_path}")
    
    logger.info("Pipeline complete!")
    return results