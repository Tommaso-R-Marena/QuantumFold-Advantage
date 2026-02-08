"""Advanced checkpoint management with best model tracking and automatic recovery.

Provides utilities for:
- Saving/loading model checkpoints with full state
- Best model tracking based on metrics
- Automatic checkpoint cleanup
- Recovery from corrupted checkpoints
- Distributed training compatibility
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""

    epoch: int
    step: int
    timestamp: str
    metrics: Dict[str, float]
    config: Dict[str, Any]
    git_commit: Optional[str] = None
    best_metric: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup and best model tracking.

    Features:
    - Automatic best model tracking
    - Configurable checkpoint retention (keep last N, keep best K)
    - Recovery from corrupted checkpoints
    - Atomic writes to prevent corruption
    - Metadata tracking for reproducibility

    Args:
        checkpoint_dir: Directory to save checkpoints
        keep_last_n: Number of recent checkpoints to keep (0 = keep all)
        keep_best_k: Number of best checkpoints to keep (0 = don't track best)
        metric_name: Metric to track for best model
        mode: 'min' or 'max' for metric optimization
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        keep_best_k: int = 2,
        metric_name: str = "val_loss",
        mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.keep_best_k = keep_best_k
        self.metric_name = metric_name
        self.mode = mode

        # Track checkpoints
        self.checkpoints: List[Dict] = []
        self._load_checkpoint_history()

    def _load_checkpoint_history(self):
        """Load checkpoint history from disk."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    self.checkpoints = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint history: {e}")
                self.checkpoints = []

    def _save_checkpoint_history(self):
        """Save checkpoint history to disk."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        try:
            with open(history_file, "w") as f:
                json.dump(self.checkpoints, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save checkpoint history: {e}")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        ema_model: Optional[torch.nn.Module] = None,
        config: Optional[Dict] = None,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint with full training state.

        Args:
            model: The model to save
            optimizer: Optimizer state
            epoch: Current epoch
            step: Current training step
            metrics: Current metrics
            scheduler: Optional learning rate scheduler
            scaler: Optional gradient scaler for mixed precision
            ema_model: Optional EMA model
            config: Optional configuration dictionary
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        if ema_model is not None:
            checkpoint["ema_model_state_dict"] = ema_model.state_dict()

        if config is not None:
            checkpoint["config"] = config

        try:
            # Get git commit if available
            import subprocess

            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
            checkpoint["git_commit"] = git_commit
        except Exception:
            pass

        # Generate checkpoint filename
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        temp_path = checkpoint_path.with_suffix(".pt.tmp")

        # Atomic write: save to temp file first
        try:
            torch.save(checkpoint, temp_path)
            # Rename only if save succeeded
            temp_path.rename(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

        # Update checkpoint history
        checkpoint_info = {
            "path": str(checkpoint_path),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "timestamp": checkpoint["timestamp"],
            "is_best": is_best,
        }
        self.checkpoints.append(checkpoint_info)
        self._save_checkpoint_history()

        # Save as best if applicable
        if is_best or self._is_best_checkpoint(metrics):
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy2(checkpoint_path, best_path)
            logger.info(f"Saved best model with {self.metric_name}={metrics.get(self.metric_name)}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        return checkpoint_path

    def _is_best_checkpoint(self, metrics: Dict[str, float]) -> bool:
        """Check if current checkpoint is the best so far."""
        if self.metric_name not in metrics:
            return False

        current_value = metrics[self.metric_name]

        # Find best previous value
        previous_values = [
            ckpt["metrics"].get(self.metric_name)
            for ckpt in self.checkpoints
            if self.metric_name in ckpt["metrics"]
        ]

        if not previous_values:
            return True

        best_previous = min(previous_values) if self.mode == "min" else max(previous_values)

        if self.mode == "min":
            return current_value < best_previous
        else:
            return current_value > best_previous

    def _cleanup_checkpoints(self):
        """Remove old checkpoints according to retention policy."""
        if self.keep_last_n == 0 and self.keep_best_k == 0:
            return  # Keep all checkpoints

        # Sort by timestamp (newest first)
        sorted_ckpts = sorted(self.checkpoints, key=lambda x: x["timestamp"], reverse=True)

        # Determine which to keep
        to_keep = set()

        # Keep last N
        if self.keep_last_n > 0:
            for ckpt in sorted_ckpts[: self.keep_last_n]:
                to_keep.add(ckpt["path"])

        # Keep best K
        if self.keep_best_k > 0 and self.metric_name:
            # Sort by metric
            metric_sorted = sorted(
                [c for c in self.checkpoints if self.metric_name in c["metrics"]],
                key=lambda x: x["metrics"][self.metric_name],
                reverse=(self.mode == "max"),
            )
            for ckpt in metric_sorted[: self.keep_best_k]:
                to_keep.add(ckpt["path"])

        # Keep explicitly marked best
        for ckpt in self.checkpoints:
            if ckpt.get("is_best", False):
                to_keep.add(ckpt["path"])

        # Remove checkpoints not in keep list
        updated_checkpoints = []
        for ckpt in self.checkpoints:
            ckpt_path = Path(ckpt["path"])
            if ckpt["path"] in to_keep:
                updated_checkpoints.append(ckpt)
            elif ckpt_path.exists():
                try:
                    ckpt_path.unlink()
                    logger.info(f"Removed old checkpoint: {ckpt_path}")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint {ckpt_path}: {e}")

        self.checkpoints = updated_checkpoints
        self._save_checkpoint_history()

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        ema_model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint and restore training state.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            scaler: Optional gradient scaler to restore
            ema_model: Optional EMA model to restore
            checkpoint_path: Specific checkpoint to load (None = latest)
            load_best: Load best model instead of latest
            strict: Strict state dict loading

        Returns:
            Dictionary with checkpoint metadata
        """
        # Determine which checkpoint to load
        if checkpoint_path:
            ckpt_path = Path(checkpoint_path)
        elif load_best:
            ckpt_path = self.checkpoint_dir / "best_model.pt"
        else:
            # Load latest checkpoint
            if not self.checkpoints:
                raise FileNotFoundError("No checkpoints found")
            ckpt_path = Path(self.checkpoints[-1]["path"])

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Load checkpoint with error handling
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
            # Try to recover from backup
            backup_path = ckpt_path.with_suffix(".pt.backup")
            if backup_path.exists():
                logger.info(f"Attempting to load from backup: {backup_path}")
                checkpoint = torch.load(backup_path, map_location="cpu")
            else:
                raise

        # Restore model state
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            if strict:
                raise

        # Restore optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")

        # Restore scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not restore scheduler state: {e}")

        # Restore scaler state
        if scaler is not None and "scaler_state_dict" in checkpoint:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not restore scaler state: {e}")

        # Restore EMA model
        if ema_model is not None and "ema_model_state_dict" in checkpoint:
            try:
                ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            except Exception as e:
                logger.warning(f"Could not restore EMA model state: {e}")

        logger.info(
            f"Loaded checkpoint from {ckpt_path} "
            f"(epoch {checkpoint['epoch']}, step {checkpoint['step']})"
        )

        return {
            "epoch": checkpoint["epoch"],
            "step": checkpoint["step"],
            "metrics": checkpoint.get("metrics", {}),
            "config": checkpoint.get("config", {}),
            "timestamp": checkpoint.get("timestamp"),
            "git_commit": checkpoint.get("git_commit"),
        }

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.checkpoints:
            return None
        return Path(self.checkpoints[-1]["path"])

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        return best_path if best_path.exists() else None
