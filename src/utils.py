"""
Utility functions for the diabetes VAE project.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json
import yaml


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model: torch.nn.Module,
               path: str,
               metadata: Optional[Dict[str, Any]] = None):
    """Save model state and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {'model_state': model.state_dict()}
    if metadata:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model: torch.nn.Module,
               path: str,
               device: str = 'cpu') -> Optional[Dict[str, Any]]:
    """Load model state and metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    
    return checkpoint.get('metadata', {})


def save_config(config, path: str):
    """Save configuration to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f)
    
    print(f"Config saved to {path}")


def load_config(path: str):
    """Load configuration from YAML."""
    from .config import Config
    return Config.from_yaml(path)


def create_experiment_dir(base_dir: str = 'experiments',
                         name: Optional[str] = None) -> Path:
    """Create experiment directory with timestamp."""
    from datetime import datetime
    
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = Path(base_dir) / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def print_model_info(model: torch.nn.Module):
    """Print model architecture and parameter count."""
    print("\n" + "=" * 50)
    print("Model Architecture:")
    print("=" * 50)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50 + "\n")


def get_device() -> str:
    """Get available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    return device


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
