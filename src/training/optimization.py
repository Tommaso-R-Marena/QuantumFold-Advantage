"""Advanced optimization utilities.

Provides:
- Custom optimizers
- Learning rate schedulers
- Gradient clipping
- Mixed precision training
- Optimization strategies
"""

import math
import warnings
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """Linear learning rate warmup."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class ExponentialWarmup(_LRScheduler):
    """Exponential warmup schedule."""

    def __init__(
        self, optimizer: Optimizer, warmup_steps: int, gamma: float = 0.99, last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Exponential warmup
            alpha = (self.last_epoch / self.warmup_steps) ** 2
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Exponential decay
            decay = self.gamma ** (self.last_epoch - self.warmup_steps)
            return [base_lr * decay for base_lr in self.base_lrs]


class GradientClipper:
    """Gradient clipping utilities."""

    @staticmethod
    def clip_grad_norm(
        parameters: Iterable[nn.Parameter], max_norm: float, norm_type: float = 2.0
    ) -> float:
        """Clip gradient norm.

        Args:
            parameters: Model parameters
            max_norm: Maximum gradient norm
            norm_type: Type of norm (default: 2.0 for L2)

        Returns:
            Total norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    @staticmethod
    def clip_grad_value(parameters: Iterable[nn.Parameter], clip_value: float):
        """Clip gradient values.

        Args:
            parameters: Model parameters
            clip_value: Maximum absolute gradient value
        """
        torch.nn.utils.clip_grad_value_(parameters, clip_value)


class MixedPrecisionTrainer:
    """Mixed precision training utilities."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()

        if self.enabled:
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except AttributeError:
                warnings.warn(
                    "Mixed precision training not available. " "Upgrade to PyTorch >= 1.6"
                )
                self.enabled = False

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: Optimizer):
        """Optimizer step with gradient scaling."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def backward(self, loss: torch.Tensor):
        """Backward pass with automatic scaling."""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()


class OptimizerFactory:
    """Factory for creating optimizers with best practices."""

    @staticmethod
    def create_adam(
        parameters: Iterable[nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> torch.optim.Adam:
        """Create Adam optimizer."""
        return torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    @staticmethod
    def create_adamw(
        parameters: Iterable[nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> torch.optim.AdamW:
        """Create AdamW optimizer (recommended)."""
        return torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    @staticmethod
    def create_lamb(
        parameters: Iterable[nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """Create LAMB optimizer (if available)."""
        try:
            from torch_optimizer import Lamb

            return Lamb(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        except ImportError:
            warnings.warn(
                "LAMB optimizer not available. Install torch-optimizer. " "Falling back to AdamW."
            )
            return OptimizerFactory.create_adamw(parameters, lr, weight_decay, betas, eps)


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_names: Tuple[str, ...] = ("bias", "LayerNorm", "layer_norm"),
) -> list:
    """Create parameter groups with differential weight decay.

    Args:
        model: PyTorch model
        weight_decay: Weight decay for most parameters
        no_decay_names: Parameter names that should not have weight decay

    Returns:
        List of parameter groups for optimizer
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
