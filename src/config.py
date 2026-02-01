"""Configuration management using Hydra.

Provides hierarchical configuration with:
- YAML-based configs
- Command-line overrides
- Type validation
- Default values
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int = 1280
    c_s: int = 384
    c_z: int = 128
    n_structure_layers: int = 8
    n_heads: int = 12
    n_query_points: int = 4
    n_point_values: int = 8
    use_quantum: bool = True
    quantum_n_qubits: int = 8
    quantum_depth: int = 4
    dropout: float = 0.1


@dataclass
class QuantumConfig:
    """Quantum circuit configuration."""
    n_qubits: int = 8
    n_layers: int = 4
    entanglement: str = "circular"
    init_strategy: str = "haar"
    rotation_gates: List[str] = field(default_factory=lambda: ["RX", "RY", "RZ"])
    add_noise: bool = False
    noise_strength: float = 0.01
    device_name: str = "default.qubit"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    accumulation_steps: int = 1
    mixed_precision_dtype: str = "float16"
    

@dataclass
class LossConfig:
    """Loss function configuration."""
    fape_weight: float = 1.0
    local_geometry_weight: float = 0.5
    perceptual_weight: float = 0.3
    confidence_weight: float = 0.1
    clash_weight: float = 0.05
    fape_clamp_distance: float = 10.0
    epsilon: float = 1e-8


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./cache")
    max_seq_length: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    use_augmentation: bool = True
    rotation_prob: float = 0.5
    noise_std: float = 0.1


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    experiment_name: str = "quantumfold_experiment"
    run_name: Optional[str] = None
    project_name: str = "quantumfold-advantage"
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500
    

@dataclass
class CheckpointConfig:
    """Checkpoint management configuration."""
    checkpoint_dir: Path = Path("./checkpoints")
    save_top_k: int = 3
    monitor_metric: str = "val_tm_score"
    mode: str = "max"
    save_last: bool = True
    auto_resume: bool = True


@dataclass
class Config:
    """Main configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Global settings
    seed: int = 42
    device: str = "cuda"
    distributed: bool = False
    num_gpus: int = 1
    debug: bool = False
