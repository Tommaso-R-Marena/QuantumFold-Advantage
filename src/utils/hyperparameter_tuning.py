"""Hyperparameter tuning utilities with Optuna integration.

Provides:
- Automated hyperparameter search
- Bayesian optimization
- Pruning of unpromising trials
- Multi-objective optimization
- Result visualization and analysis
"""

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler
import torch
import logging
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Automated hyperparameter tuning with Optuna.
    
    Supports:
    - Bayesian optimization (TPE, CMA-ES)
    - Early stopping (MedianPruner, HyperbandPruner)
    - Parallel trials
    - Study resumption
    - Result analysis and visualization
    
    Args:
        study_name: Name for the optimization study
        storage: Database URL for study persistence (e.g., 'sqlite:///optuna.db')
        direction: 'minimize' or 'maximize'
        sampler_name: 'tpe' or 'cmaes'
        pruner_name: 'median' or 'hyperband'
    """
    
    def __init__(
        self,
        study_name: str = "quantumfold_tuning",
        storage: Optional[str] = None,
        direction: str = "minimize",
        sampler_name: str = "tpe",
        pruner_name: str = "median",
        n_startup_trials: int = 10
    ):
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        
        # Initialize sampler
        if sampler_name == "tpe":
            self.sampler = TPESampler(n_startup_trials=n_startup_trials)
        elif sampler_name == "cmaes":
            self.sampler = CmaEsSampler(n_startup_trials=n_startup_trials)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        # Initialize pruner
        if pruner_name == "median":
            self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner_name == "hyperband":
            self.pruner = HyperbandPruner()
        else:
            self.pruner = None
        
        # Create or load study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        logger.info(
            f"Initialized tuner: {study_name}, "
            f"direction={direction}, sampler={sampler_name}"
        )
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial.
        
        Override this method to customize the search space.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            # Model architecture
            'hidden_dim': trial.suggest_categorical(
                'hidden_dim', [256, 384, 512, 768, 1024]
            ),
            'num_layers': trial.suggest_int('num_layers', 4, 16),
            'num_heads': trial.suggest_categorical('num_heads', [4, 8, 12, 16]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            
            # Quantum parameters
            'num_qubits': trial.suggest_int('num_qubits', 4, 12),
            'num_quantum_layers': trial.suggest_int('num_quantum_layers', 1, 4),
            
            # Training
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            
            # Optimizer
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
            'scheduler': trial.suggest_categorical(
                'scheduler', ['cosine', 'linear', 'exponential']
            ),
        }
        
        return params
    
    def objective(
        self,
        trial: optuna.Trial,
        train_fn: Callable,
        config: Optional[Dict] = None
    ) -> float:
        """Objective function for optimization.
        
        Args:
            trial: Optuna trial
            train_fn: Training function that returns validation metric
            config: Base configuration dictionary
            
        Returns:
            Validation metric value
        """
        # Get suggested hyperparameters
        params = self.suggest_hyperparameters(trial)
        
        # Merge with base config
        if config is not None:
            params = {**config, **params}
        
        try:
            # Run training
            metric = train_fn(params, trial)
            
            # Report intermediate value for pruning
            trial.report(metric, step=trial.number)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return metric
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return worst possible value
            return float('inf') if self.direction == 'minimize' else float('-inf')
    
    def optimize(
        self,
        train_fn: Callable,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        config: Optional[Dict] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True
    ) -> optuna.Study:
        """Run hyperparameter optimization.
        
        Args:
            train_fn: Training function
            n_trials: Number of trials to run
            timeout: Time limit in seconds
            config: Base configuration
            n_jobs: Number of parallel jobs
            show_progress_bar: Whether to show progress
            
        Returns:
            Completed Optuna study
        """
        logger.info(f"Starting optimization with {n_trials} trials")
        
        self.study.optimize(
            lambda trial: self.objective(trial, train_fn, config),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar
        )
        
        logger.info("Optimization complete")
        self.print_best_results()
        
        return self.study
    
    def print_best_results(self):
        """Print best trial results."""
        best_trial = self.study.best_trial
        
        logger.info("=" * 50)
        logger.info("Best Trial Results:")
        logger.info(f"  Value: {best_trial.value:.6f}")
        logger.info(f"  Trial number: {best_trial.number}")
        logger.info("  Hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
        logger.info("=" * 50)
    
    def save_best_params(self, filepath: str):
        """Save best hyperparameters to JSON file.
        
        Args:
            filepath: Path to save parameters
        """
        best_params = self.study.best_params
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"Saved best parameters to {filepath}")
    
    def get_importance(self) -> Dict[str, float]:
        """Calculate parameter importance.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        try:
            importance = optuna.importance.get_param_importances(self.study)
            
            logger.info("Parameter Importance:")
            for param, score in sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            ):
                logger.info(f"  {param}: {score:.4f}")
            
            return importance
        except Exception as e:
            logger.warning(f"Could not calculate importance: {e}")
            return {}
    
    def plot_optimization_history(self, filepath: Optional[str] = None):
        """Plot optimization history.
        
        Args:
            filepath: Path to save plot (None to show)
        """
        try:
            from optuna.visualization import plot_optimization_history
            import plotly.io as pio
            
            fig = plot_optimization_history(self.study)
            
            if filepath:
                pio.write_image(fig, filepath)
                logger.info(f"Saved optimization history to {filepath}")
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not installed, cannot create plots")
    
    def plot_param_importances(self, filepath: Optional[str] = None):
        """Plot parameter importances.
        
        Args:
            filepath: Path to save plot (None to show)
        """
        try:
            from optuna.visualization import plot_param_importances
            import plotly.io as pio
            
            fig = plot_param_importances(self.study)
            
            if filepath:
                pio.write_image(fig, filepath)
                logger.info(f"Saved parameter importances to {filepath}")
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not installed, cannot create plots")


def create_training_objective(
    model_class,
    dataset,
    n_epochs: int = 10,
    device: str = 'cuda'
) -> Callable:
    """Create a training objective function for hyperparameter tuning.
    
    Args:
        model_class: Model class to instantiate
        dataset: Training dataset
        n_epochs: Number of epochs to train
        device: Device to train on
        
    Returns:
        Objective function
    """
    def objective(params: Dict[str, Any], trial: Optional[optuna.Trial] = None) -> float:
        """Train model and return validation loss."""
        # Create model with suggested hyperparameters
        model = model_class(**params).to(device)
        
        # Create optimizer
        if params.get('optimizer', 'adam') == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params.get('weight_decay', 0)
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params.get('weight_decay', 0)
            )
        
        # Training loop
        model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for batch in dataset:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataset)
            
            # Report intermediate value
            if trial is not None:
                trial.report(avg_loss, epoch)
                
                # Prune if necessary
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        return avg_loss
    
    return objective
