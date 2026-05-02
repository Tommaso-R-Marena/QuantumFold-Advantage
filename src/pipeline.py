"""Unified training and evaluation pipeline."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import load_or_generate_data
from .models import UnifiedQuantumFold
from .training import AdvancedTrainer, TrainingConfig
from .utils import set_seed, save_pdb
from .auto_pipeline import AutoImprovementEngine

logger = logging.getLogger(__name__)

def run_unified_pipeline(
    modality: str = "protein",
    use_quantum: bool = True,
    device: str = "auto",
    epochs: int = 10,
    batch_size: int = 8,
    seed: int = 42,
    output_dir: str = "outputs",
    auto_improve: bool = True
):
    """
    Run end-to-end unified training pipeline.
    """
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info(f"Unified Pipeline: modality={modality}, quantum={use_quantum}, device={device}")

    # 1. Load Data
    train_loader, val_loader = load_or_generate_data(
        num_synthetic=100, seed=seed
    )

    # 2. Build Unified Model
    model = UnifiedQuantumFold(
        use_quantum=use_quantum
    ).to(device)

    # 3. Setup Advanced Trainer
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir=str(output_path / "checkpoints"),
        use_amp=(device.type == "cuda")
    )

    trainer = AdvancedTrainer(
        model=model,
        config=config,
        device=device
    )

    # 4. Training with Auto-Improvement
    engine = AutoImprovementEngine()
    current_settings = {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay
    }

    logger.info("Starting unified training...")
    history = trainer.train(train_loader, val_loader)

    if auto_improve:
        # Simple extraction of last loss values for auto-improvement suggestion
        hist_dict = {
            "train_loss": [h["total"] for h in history["train"]],
            "val_loss": [h["total"] for h in history["val"]]
        }
        suggestions = engine.suggest(hist_dict, current_settings)
        with open(output_path / "improved_settings.json", "w") as f:
            json.dump(suggestions, f, indent=2)
        logger.info(f"Auto-improvement suggestions saved to {output_path / 'improved_settings.json'}")

    # 5. Final Evaluation & Visualization
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        # Handle legacy batch format from generate_synthetic_data
        if isinstance(batch, dict):
            # This is already handled in trainer.train_epoch, but for manual viz:
            seq = batch["sequence"].to(device)
        else:
            seq, _ = batch
            seq = seq.to(device)

        out = model(protein_embeddings=seq) # Simplified for demo
        coords = out["coordinates"][0].cpu().numpy()

        pdb_file = output_path / f"final_{modality}.pdb"
        save_pdb(coords, "A" * len(coords), str(pdb_file))
        logger.info(f"Final structure saved to {pdb_file}")

    return history

def run_pipeline(*args, **kwargs):
    """Legacy compatibility wrapper."""
    # Map old args to new ones
    mode = kwargs.get("mode", "classical")
    use_quantum = (mode == "hybrid")
    return run_unified_pipeline(
        use_quantum=use_quantum,
        epochs=kwargs.get("epochs", 10),
        device=kwargs.get("device", "cpu"),
        output_dir=kwargs.get("output_dir", "outputs")
    )
