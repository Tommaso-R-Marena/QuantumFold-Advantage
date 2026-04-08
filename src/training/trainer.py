"""Training loop for QuantumFold-Advantage models.

Supports both quantum-enabled and classical-only modes, with separate
learning rates for quantum and classical parameters, gradient clipping,
cosine-annealing scheduler, early stopping, and checkpoint management.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_rmsd, compute_tm_score, kabsch_align
from src.models.quantumfold_advantage import QuantumFoldAdvantage
from src.training.losses import CombinedLoss

logger = logging.getLogger(__name__)


class QuantumFoldTrainer:
    """Trainer for QuantumFold-Advantage.

    Args:
        model: QuantumFoldAdvantage model instance.
        lr: Learning rate for classical parameters.
        quantum_lr: Learning rate for quantum parameters (typically higher).
        weight_decay: AdamW weight decay.
        grad_clip: Max gradient norm.
        patience: Early stopping patience (epochs).
        checkpoint_dir: Directory for saving checkpoints.
    """

    def __init__(
        self,
        model: QuantumFoldAdvantage,
        lr: float = 1e-4,
        quantum_lr: float = 1e-2,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        patience: int = 15,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.grad_clip = grad_clip
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Separate parameter groups
        quantum_params = []
        classical_params = []
        for name, param in model.named_parameters():
            if "quantum" in name:
                quantum_params.append(param)
            else:
                classical_params.append(param)

        param_groups = [{"params": classical_params, "lr": lr}]
        if quantum_params:
            param_groups.append({"params": quantum_params, "lr": quantum_lr})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        self.criterion = CombinedLoss()

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_rmsd": [],
            "val_tm_score": [],
        }
        self.best_val_tm = -1.0
        self.epochs_no_improve = 0

    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0

        for batch in loader:
            aa_idx = batch["aa_idx"].to(self.device)
            physchem = batch["physchem"].to(self.device)
            coords_true = batch["coords"].to(self.device)
            mask = batch["mask"].to(self.device)

            pred = self.model(aa_idx, physchem, mask=mask)

            # For loss, we need true rotations/translations — use identity
            B, L = aa_idx.shape
            true_rot = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
            true_trans = coords_true[:, :, 1, :]  # Cα as translation

            loss = self.criterion(pred, coords_true, true_rot, true_trans, mask)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * aa_idx.size(0)
            n += aa_idx.size(0)

        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_rmsd: List[float] = []
        all_tm: List[float] = []
        n = 0

        for batch in loader:
            aa_idx = batch["aa_idx"].to(self.device)
            physchem = batch["physchem"].to(self.device)
            coords_true = batch["coords"].to(self.device)
            mask = batch["mask"].to(self.device)

            pred = self.model(aa_idx, physchem, mask=mask)

            B, L = aa_idx.shape
            true_rot = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
            true_trans = coords_true[:, :, 1, :]

            loss = self.criterion(pred, coords_true, true_rot, true_trans, mask)
            total_loss += loss.item() * B
            n += B

            # Per-sample metrics
            for i in range(B):
                length = int(mask[i].sum().item())
                p = pred["coords_ca"][i, :length].cpu().numpy()
                t = coords_true[i, :length, 1, :].cpu().numpy()
                all_rmsd.append(compute_rmsd(p, t))
                all_tm.append(compute_tm_score(p, t))

        return {
            "val_loss": total_loss / max(n, 1),
            "val_rmsd": float(np.mean(all_rmsd)),
            "val_tm_score": float(np.mean(all_tm)),
        }

    # ------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        scheduler_type: str = "cosine",
    ) -> Dict[str, List[float]]:
        """Run full training loop.

        Args:
            train_loader: Training data.
            val_loader: Validation data.
            epochs: Maximum epochs.
            scheduler_type: 'cosine' or 'step'.

        Returns:
            Training history dict.
        """
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=max(epochs // 4, 5), T_mult=2
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)

        mode_tag = "QUANTUM" if self.model.quantum_enabled else "CLASSICAL"
        logger.info(f"Training [{mode_tag}] for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            scheduler.step()

            self.history["train_loss"].append(train_loss)
            for k, v in val_metrics.items():
                self.history.setdefault(k, []).append(v)

            elapsed = time.time() - t0
            logger.info(
                f"[{mode_tag}] Epoch {epoch:03d}/{epochs} | "
                f"Train Loss {train_loss:.4f} | "
                f"Val Loss {val_metrics['val_loss']:.4f} | "
                f"RMSD {val_metrics['val_rmsd']:.3f} | "
                f"TM {val_metrics['val_tm_score']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Checkpointing
            if val_metrics["val_tm_score"] > self.best_val_tm:
                self.best_val_tm = val_metrics["val_tm_score"]
                self.epochs_no_improve = 0
                self._save_checkpoint(f"best_{mode_tag.lower()}.pt")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping after {epoch} epochs")
                break

        self._save_checkpoint(f"final_{mode_tag.lower()}.pt")
        return self.history

    # ------------------------------------------------------------------
    def _save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "best_val_tm": self.best_val_tm,
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")
