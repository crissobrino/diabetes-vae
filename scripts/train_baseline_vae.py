"""
Train baseline VAE on CGM data only.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from torch.utils.data import DataLoader, random_split
from config import Config
from dataset import create_dummy_dataset
from models import BaselineVAE
from training import VAETrainer
from utils import set_seed, print_model_info, get_device, create_experiment_dir


def main():
    # Configuration
    config = Config()
    set_seed(42)
    device = get_device()
    
    # Create experiment directory
    exp_dir = create_experiment_dir(base_dir='experiments', name='baseline_vae')
    config.training.save_dir = str(exp_dir / 'models')
    
    print(f"Experiment directory: {exp_dir}")
    
    # Create dummy dataset (replace with real data loading)
    print("Loading dataset...")
    dataset = create_dummy_dataset(n_samples=200)
    
    # Split into train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.training.batch_size, shuffle=False)
    
    print(f"Train set: {len(train_set)} samples")
    print(f"Val set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")
    
    # Create model
    print("\nCreating baseline VAE...")
    model = BaselineVAE(
        cgm_length=config.data.cgm_readings_per_day,
        encoder_channels=config.model.cgm_encoder_channels,
        decoder_channels=config.model.cgm_decoder_channels,
        latent_dim=config.model.latent_dim
    )
    
    print_model_info(model)
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.num_epochs,
        kl_annealing=config.training.kl_annealing_enabled,
        warmup_epochs=config.training.kl_warmup_epochs,
        save_dir=config.training.save_dir,
        save_freq=config.training.checkpoint_freq
    )
    
    # Save training history
    history_path = exp_dir / 'training_history.json'
    trainer.save_history(str(history_path))
    print(f"\nTraining history saved to {history_path}")
    
    # Save config
    config_path = exp_dir / 'config.yaml'
    config.save_yaml(str(config_path))
    print(f"Config saved to {config_path}")
    
    print(f"\nTraining complete! Results saved to {exp_dir}")


if __name__ == '__main__':
    main()
