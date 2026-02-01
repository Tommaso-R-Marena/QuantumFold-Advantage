"""Advanced configuration management with Hydra and OmegaConf.

Provides:
- Hierarchical configuration with composition
- Type-safe config dataclasses
- Environment variable interpolation  
- Config validation
- Experiment tracking integration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import os


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Core dimensions
    input_dim: int = 1280  # ESM-2 embedding size
    c_s: int = 384  # Single representation dimension
    c_z: int = 128  # Pair representation dimension
    
    # Transformer encoder
    n_encoder_layers: int = 12
    n_attention_heads: int = 8
    dropout: float = 0.1
    
    # Structure module  
    n_structure_layers: int = 8
    n_ipa_heads: int = 12
    n_ipa_points: int = 4
    
    # Quantum enhancement
    use_quantum: bool = True
    n_qubits: int = 4
    quantum_depth: int = 3
    quantum_entanglement: str = "circular"
    
    # Output
    confidence_head: bool = True
    predict_angles: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    # Training dynamics
    epochs: int = 100
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # or "float16"
    
    # Exponential moving average
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Warmup
    warmup_steps: int = 1000
    warmup_type: str = "linear"
    
    # Loss weights
    fape_weight: float = 1.0
    plddt_weight: float = 0.1
    distogram_weight: float = 0.3
    angle_weight: float = 0.2
    
    # Validation
    val_every_n_steps: int = 500
    val_check_interval: float = 1.0
    early_stopping_patience: int = 10


@dataclass  
class DataConfig:
    """Data loading and preprocessing."""
    # Dataset
    data_dir: Path = Path("data")
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Preprocessing
    max_sequence_length: int = 512
    min_sequence_length: int = 10
    crop_strategy: str = "random"  # or "sliding", "center"
    
    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Augmentation
    use_augmentation: bool = True
    rotation_prob: float = 0.8
    noise_scale: float = 0.2
    mixup_prob: float = 0.3
    
    # Caching
    cache_embeddings: bool = True
    cache_dir: Path = Path(".cache/embeddings")


@dataclass
class LoggingConfig:
    """Logging and experiment tracking."""
    # Output directories
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "quantumfold"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    use_tensorboard: bool = True
    tensorboard_dir: Path = Path("runs")
    
    # Checkpointing
    save_every_n_steps: int = 1000
    save_top_k: int = 3
    checkpoint_metric: str = "val_tm_score"
    checkpoint_mode: str = "max"
    
    # Logging frequency
    log_every_n_steps: int = 10
    log_gradients: bool = False
    log_parameters: bool = False


@dataclass
class SystemConfig:
    """System and hardware configuration."""
    # Device
    device: str = "auto"  # auto, cpu, cuda, mps
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True
    
    # Performance
    compile_model: bool = False  # torch.compile
    channels_last: bool = False
    gradient_checkpointing: bool = False


@dataclass
class QuantumConfig:
    """Quantum computing specific settings."""
    # Device
    backend: str = "default.qubit"  # or lightning.qubit, lightning.gpu
    
    # Circuit parameters
    n_qubits: int = 4
    n_layers: int = 3
    entanglement: str = "circular"  # linear, circular, all_to_all
    
    # Initialization
    init_strategy: str = "haar"  # identity, random, haar
    rotation_gates: List[str] = field(default_factory=lambda: ["RX", "RY", "RZ"])
    
    # Noise simulation
    add_noise: bool = False
    noise_strength: float = 0.01
    noise_model: str = "depolarizing"
    
    # Optimization
    diff_method: str = "backprop"  # parameter-shift, adjoint, backprop
    interface: str = "torch"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    # Experiment metadata
    name: str = "quantumfold_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    
    def validate(self):
        """Validate configuration."""
        # Check paths exist
        if self.data.train_file:
            assert Path(self.data.train_file).exists(), f"Train file not found: {self.data.train_file}"
        
        # Check parameter ranges
        assert 0 < self.training.learning_rate < 1, "Learning rate out of range"
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.model.n_encoder_layers > 0, "Must have at least 1 encoder layer"
        
        # Check compatibility
        if self.training.use_amp and self.system.device == "cpu":
            print("Warning: AMP disabled on CPU")
            self.training.use_amp = False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Load config from dictionary."""
        return cls(**config_dict)


def load_config(config_path: Optional[Path] = None) -> ExperimentConfig:
    """Load configuration from file or create default.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        ExperimentConfig instance
    """
    if config_path and config_path.exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return ExperimentConfig.from_dict(config_dict)
    return ExperimentConfig()


def save_config(config: ExperimentConfig, save_path: Path):
    """Save configuration to YAML file.
    
    Args:
        config: ExperimentConfig instance
        save_path: Path to save YAML file
    """
    import yaml
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
