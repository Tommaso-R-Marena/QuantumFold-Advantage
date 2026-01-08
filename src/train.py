"""
Training loop and evaluation metrics for protein structure prediction.

Key functions:
- train_epoch: Single training epoch
- evaluate_model: Compute RMSD and TM-score on test set
- compute_rmsd: Root-mean-square deviation between coordinate sets
- compute_tm_score: Template modeling score (topology-aware similarity)

References:
    - RMSD and Kabsch algorithm: Kabsch (1976), DOI: 10.1107/S0567739476001873
    - TM-score: Zhang & Skolnick (2005), DOI: 10.1093/nar/gki524
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_rmsd(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
    """
    Compute RMSD between predicted and true coordinates.
    
    Args:
        pred_coords: Predicted coordinates (N, 3)
        true_coords: True coordinates (N, 3)
    
    Returns:
        RMSD in Angstroms
    
    Notes:
        Does NOT apply Kabsch superposition. For aligned RMSD, use
        scripts/statistical_evaluation.py which implements Kabsch algorithm.
    
    References:
        RMSD definition: sqrt(mean(||pred - true||^2))
    """
    assert pred_coords.shape == true_coords.shape, "Coordinate shape mismatch"
    diff = pred_coords - true_coords
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return float(rmsd)


def compute_tm_score(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
    """
    Compute simplified TM-score.
    
    Args:
        pred_coords: Predicted coordinates (N, 3)
        true_coords: True coordinates (N, 3)
    
    Returns:
        TM-score (0-1, higher is better)
    
    Notes:
        This is a SIMPLIFIED version for demonstration. Full TM-score requires
        iterative optimization to find best superposition. See TM-align
        (Zhang & Skolnick 2005, DOI: 10.1093/nar/gki524) for details.
        
        For rigorous evaluation, use scripts/statistical_evaluation.py which
        implements proper Kabsch superposition and LG-score iteration.
    
    Formula (simplified):
        TM-score ≈ (1/N) * Σ [1 / (1 + (d_i / d0)^2)]
        where d_i = distance between residue i, d0 = 1.24 * N^(1/3) - 1.8
    """
    N = len(pred_coords)
    d0 = 1.24 * (N ** (1.0 / 3.0)) - 1.8
    
    distances = np.sqrt(np.sum((pred_coords - true_coords) ** 2, axis=1))
    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    tm_score = np.mean(scores)
    
    return float(tm_score)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer (e.g., Adam)
        device: Compute device
        epoch: Current epoch number (for logging)
    
    Returns:
        Dictionary with training metrics (loss, rmsd)
    """
    model.train()
    total_loss = 0.0
    total_rmsd = 0.0
    n_batches = 0
    
    criterion = nn.MSELoss()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        sequences = batch["sequence"].to(device)
        coords_true = batch["coordinates"].to(device)
        
        # Forward pass
        coords_pred = model(sequences)
        
        # Compute loss
        loss = criterion(coords_pred, coords_true)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        
        # Compute RMSD for first sample in batch
        with torch.no_grad():
            pred_np = coords_pred[0].cpu().numpy()
            true_np = coords_true[0].cpu().numpy()
            rmsd = compute_rmsd(pred_np, true_np)
            total_rmsd += rmsd
        
        n_batches += 1
        pbar.set_postfix({"loss": loss.item(), "rmsd": rmsd})
    
    avg_loss = total_loss / n_batches
    avg_rmsd = total_rmsd / n_batches
    
    return {"loss": avg_loss, "rmsd": avg_rmsd}


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Compute device
    
    Returns:
        Dictionary with evaluation metrics (loss, rmsd, tm_score)
    """
    model.eval()
    total_loss = 0.0
    total_rmsd = 0.0
    total_tm_score = 0.0
    n_samples = 0
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences = batch["sequence"].to(device)
            coords_true = batch["coordinates"].to(device)
            
            # Forward pass
            coords_pred = model(sequences)
            
            # Loss
            loss = criterion(coords_pred, coords_true)
            total_loss += loss.item() * len(sequences)
            
            # Compute metrics for each sample
            for i in range(len(sequences)):
                pred_np = coords_pred[i].cpu().numpy()
                true_np = coords_true[i].cpu().numpy()
                
                rmsd = compute_rmsd(pred_np, true_np)
                tm_score = compute_tm_score(pred_np, true_np)
                
                total_rmsd += rmsd
                total_tm_score += tm_score
                n_samples += 1
    
    results = {
        "loss": total_loss / n_samples,
        "rmsd": total_rmsd / n_samples,
        "tm_score": total_tm_score / n_samples,
    }
    
    return results


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 0.001
) -> Tuple[nn.Module, Dict]:
    """
    Full training loop.
    
    Args:
        model: Model to train
        train_loader: Training data
        test_loader: Test data
        device: Compute device
        epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        (trained_model, metrics_dict) tuple
    """
    logger = logging.getLogger(__name__)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "train_rmsd": [],
        "test_loss": [],
        "test_rmsd": [],
        "test_tm_score": [],
    }
    
    logger.info(f"Starting training: {epochs} epochs, lr={lr}")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_rmsd"].append(train_metrics["rmsd"])
        
        # Evaluate
        test_metrics = evaluate_model(model, test_loader, device)
        history["test_loss"].append(test_metrics["loss"])
        history["test_rmsd"].append(test_metrics["rmsd"])
        history["test_tm_score"].append(test_metrics["tm_score"])
        
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f}, RMSD: {train_metrics['rmsd']:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f}, RMSD: {test_metrics['rmsd']:.4f}, "
            f"TM-Score: {test_metrics['tm_score']:.4f}"
        )
    
    return model, history
