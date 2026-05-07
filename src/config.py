"""
Configuration management for the diabetes VAE project.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    cgm_resolution: int = 5  # minutes between CGM readings
    cgm_readings_per_day: int = 288  # 24 hours * 60 / 5
    dataset_split: Dict[str, float] = None
    
    def __post_init__(self):
        if self.dataset_split is None:
            self.dataset_split = {"train": 0.7, "val": 0.15, "test": 0.15}


@dataclass
class ModelConfig:
    """Configuration for VAE models."""
    latent_dim: int = 8
    cgm_seq_length: int = 288
    insulin_seq_length: int = 288
    
    # Encoder architecture
    cgm_encoder_channels: List[int] = None
    cgm_encoder_kernel_size: int = 3
    cgm_encoder_stride: int = 1
    cgm_encoder_padding: int = 1
    
    # Decoder architecture
    cgm_decoder_channels: List[int] = None
    
    # Multimodal VAE
    use_product_of_experts: bool = True
    missing_modality_handling: str = "drop"  # "drop" or "zero"
    
    def __post_init__(self):
        if self.cgm_encoder_channels is None:
            self.cgm_encoder_channels = [16, 32, 64]
        if self.cgm_decoder_channels is None:
            self.cgm_decoder_channels = [64, 32, 16]


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # KL annealing
    kl_annealing_enabled: bool = True
    kl_warmup_epochs: int = 10
    
    # Regularization
    kl_weight: float = 1.0
    reconstruction_weight: float = 1.0
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    
    # Checkpointing
    checkpoint_freq: int = 5  # epochs
    save_dir: str = "models"


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    plot_missing_modality_combos: bool = True
    reconstruction_threshold: float = 0.1


class Config:
    """Master configuration object."""
    
    def __init__(self, 
                 data_config: Optional[DataConfig] = None,
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 eval_config: Optional[EvalConfig] = None):
        self.data = data_config or DataConfig()
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        self.eval = eval_config or EvalConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        eval_config = EvalConfig(**config_dict.get('eval', {}))
        
        return cls(data_config, model_config, training_config, eval_config)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'eval': self.eval.__dict__,
        }
    
    def save_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Default configuration
DEFAULT_CONFIG = Config()
