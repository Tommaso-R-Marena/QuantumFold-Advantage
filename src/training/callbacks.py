"""Training callbacks for experiment management.

Provides:
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Metric tracking
- TensorBoard integration
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import warnings

import torch
import numpy as np


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None):
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        pass


class EarlyStopping(Callback):
    """Stop training when metric stops improving."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best_value = np.inf
        elif mode == 'max':
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best_value = -np.inf
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            warnings.warn(
                f"Early stopping requires {self.monitor} available. Skipping.",
                RuntimeWarning
            )
            return
        
        if self.monitor_op(current, self.best_value):
            self.best_value = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() 
                                    for k, v in logs.get('model_state', {}).items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logs['stop_training'] = True
                
                if self.verbose:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                    print(f"Best {self.monitor}: {self.best_value:.4f}")
                
                if self.restore_best_weights and self.best_weights:
                    logs['restore_weights'] = self.best_weights


class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min',
        save_freq: int = 1,
        verbose: bool = True
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.verbose = verbose
        
        self.epochs_since_last_save = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_value = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_value = -np.inf
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Create directory
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        
        if self.epochs_since_last_save < self.save_freq:
            return
        
        current = logs.get(self.monitor)
        
        if current is None:
            if not self.save_best_only:
                self._save_model(epoch, logs)
            return
        
        if self.save_best_only:
            if self.monitor_op(current, self.best_value):
                self.best_value = current
                self._save_model(epoch, logs)
        else:
            self._save_model(epoch, logs)
    
    def _save_model(self, epoch, logs):
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        
        model_state = logs.get('model_state', {})
        if model_state:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'metrics': {k: v for k, v in logs.items() 
                           if isinstance(v, (int, float))}
            }, filepath)
            
            self.epochs_since_last_save = 0
            
            if self.verbose:
                print(f"\n💾 Checkpoint saved: {filepath}")


class LearningRateScheduler(Callback):
    """Adjust learning rate during training."""
    
    def __init__(
        self,
        schedule: Callable[[int], float],
        verbose: bool = True
    ):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.schedule(epoch)
        logs = logs or {}
        logs['learning_rate'] = lr
        
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.6f}")


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when metric plateaus."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        mode: str = 'min',
        verbose: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose
        
        self.wait = 0
        
        if mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b)
            self.best_value = np.inf
        elif mode == 'max':
            self.monitor_op = lambda a, b: np.greater(a, b)
            self.best_value = -np.inf
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        current_lr = logs.get('learning_rate')
        
        if current is None or current_lr is None:
            return
        
        if self.monitor_op(current, self.best_value):
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                new_lr = max(current_lr * self.factor, self.min_lr)
                
                if new_lr < current_lr:
                    logs['reduce_lr'] = new_lr
                    self.wait = 0
                    
                    if self.verbose:
                        print(f"\n📉 Reducing learning rate to {new_lr:.6f}")


class MetricTracker(Callback):
    """Track and log metrics during training."""
    
    def __init__(self, metrics: List[str]):
        super().__init__()
        self.metrics = metrics
        self.history = {metric: [] for metric in metrics}
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        for metric in self.metrics:
            if metric in logs:
                self.history[metric].append(logs[metric])
    
    def get_history(self) -> Dict[str, List[float]]:
        return self.history
    
    def save_history(self, filepath: str):
        """Save metric history to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


class ProgressCallback(Callback):
    """Display training progress."""
    
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        metrics_str = ' - '.join([
            f"{k}: {v:.4f}" for k, v in logs.items()
            if isinstance(v, (int, float)) and not k.startswith('_')
        ])
        
        progress = (epoch + 1) / self.total_epochs * 100
        print(f"\rEpoch {epoch + 1}/{self.total_epochs} [{progress:>5.1f}%] - {metrics_str}", 
              end='')
        
        if epoch + 1 == self.total_epochs:
            print()  # New line at end


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
