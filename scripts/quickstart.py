"""
Quickstart script demonstrating basic usage of the diabetes VAE project.
Run this to get started with the models and understand the main workflows.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from config import Config, DataConfig, ModelConfig, TrainingConfig
from dataset import DiabeticPatientDayDataset, create_dummy_dataset
from models import BaselineVAE, MultimodalVAEWithPoE
from utils import set_seed, get_device, print_model_info


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def quickstart_basic_setup():
    """Basic setup and device configuration."""
    print_section("1. SETUP")
    
    set_seed(42)
    device = get_device()
    
    # Show configuration
    config = Config()
    print("Default configuration loaded.")
    print(f"  Latent dimension: {config.model.latent_dim}")
    print(f"  CGM readings per day: {config.data.cgm_readings_per_day}")
    print(f"  Batch size: {config.training.batch_size}")


def quickstart_data():
    """Dataset creation and inspection."""
    print_section("2. DATA LOADING")
    
    # Create synthetic dataset
    print("Creating synthetic patient-day dataset...")
    dataset = create_dummy_dataset(n_samples=100, n_physiology_features=5)
    
    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Available modalities: {dataset.modalities}")
    
    # Inspect a sample
    sample = dataset[0]
    print(f"\nSample 0 keys: {sample.keys()}")
    print(f"  CGM shape: {sample['cgm'].shape}")
    print(f"  Insulin shape: {sample['insulin'].shape}")
    if sample['physiology'] is not None:
        print(f"  Physiology shape: {sample['physiology'].shape}")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  Clinical targets: {[k for k in sample.keys() if k.startswith(('time_', 'glucose_', 'mean_'))]}")
    
    # Create data loaders
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    
    print(f"\n  Train set: {len(train_set)} samples")
    print(f"  Val set: {len(val_set)} samples")
    print(f"  Test set: {len(test_set)} samples")
    
    return train_loader, val_loader


def quickstart_baseline_vae(device):
    """Baseline VAE model creation and forward pass."""
    print_section("3. BASELINE VAE")
    
    config = Config()
    
    print("Creating baseline VAE (CGM only)...")
    model = BaselineVAE(
        cgm_length=config.data.cgm_readings_per_day,
        encoder_channels=config.model.cgm_encoder_channels,
        decoder_channels=config.model.cgm_decoder_channels,
        latent_dim=config.model.latent_dim
    )
    model.to(device)
    
    print_model_info(model)
    
    # Forward pass
    batch_cgm = torch.randn(4, 288).to(device)
    print("Forward pass with batch of 4 samples (shape: 4 x 288)...")
    
    with torch.no_grad():
        recon, mu, logvar = model(batch_cgm)
    
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Latent mean (mu) shape: {mu.shape}")
    print(f"  Latent log-var shape: {logvar.shape}")
    
    # Encode and decode
    z = model.encode(batch_cgm)
    recon_from_z = model.decode(z)
    print(f"\nEncode-decode: {batch_cgm.shape} -> {z.shape} -> {recon_from_z.shape}")
    
    return model


def quickstart_multimodal_vae(device):
    """Multimodal VAE with Product-of-Experts."""
    print_section("4. MULTIMODAL VAE WITH PRODUCT-OF-EXPERTS")
    
    config = Config()
    
    print("Creating multimodal VAE (CGM + Insulin + Physiology)...")
    model = MultimodalVAEWithPoE(
        cgm_length=config.data.cgm_readings_per_day,
        insulin_length=config.data.cgm_readings_per_day,
        physiology_dim=5,
        encoder_channels=config.model.cgm_encoder_channels,
        decoder_channels=config.model.cgm_decoder_channels,
        latent_dim=config.model.latent_dim
    )
    model.to(device)
    
    print_model_info(model)
    
    # Forward pass with all modalities
    batch_cgm = torch.randn(4, 288).to(device)
    batch_insulin = torch.randn(4, 288).to(device)
    batch_phys = torch.randn(4, 5).to(device)
    
    print("Forward pass with all modalities (batch size: 4)...")
    with torch.no_grad():
        outputs = model(cgm=batch_cgm, insulin=batch_insulin, physiology=batch_phys)
    
    print(f"  Output keys: {outputs.keys()}")
    print(f"  Latent code (z) shape: {outputs['z'].shape}")
    print(f"  CGM reconstruction shape: {outputs['cgm_recon'].shape}")
    print(f"  Insulin reconstruction shape: {outputs['insulin_recon'].shape}")
    print(f"  Physiology reconstruction shape: {outputs['physiology_recon'].shape}")
    
    # Test missing modality handling
    print("\nTesting missing modality robustness...")
    with torch.no_grad():
        # Only CGM and insulin
        outputs_no_phys = model(cgm=batch_cgm, insulin=batch_insulin, physiology=None)
        print(f"  Without physiology - Output keys: {outputs_no_phys.keys()}")
        
        # Only CGM
        outputs_cgm_only = model(cgm=batch_cgm, insulin=None, physiology=None)
        print(f"  Only CGM - Output keys: {outputs_cgm_only.keys()}")
    
    return model


def quickstart_training_loss():
    """Demonstrate VAE loss computation."""
    print_section("5. VAE LOSS COMPUTATION")
    
    from training import VAETrainer
    
    device = get_device()
    config = Config()
    
    # Create a simple model
    model = BaselineVAE(latent_dim=config.model.latent_dim)
    trainer = VAETrainer(model, device=device)
    
    # Generate dummy data
    batch_cgm = torch.randn(8, 288)
    recon = torch.randn(8, 288)
    mu = torch.randn(8, config.model.latent_dim)
    logvar = torch.randn(8, config.model.latent_dim)
    
    print("Computing VAE loss...")
    loss, recon_loss, kl_loss = trainer.vae_loss(recon, batch_cgm, mu, logvar, kl_weight=1.0)
    
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL divergence: {kl_loss.item():.4f}")
    
    # Test KL annealing
    print("\nKL annealing schedule (over 50 epochs with 10 warmup):")
    for epoch in [0, 5, 10, 15, 25, 50]:
        kl_weight = trainer.get_kl_weight(epoch, total_epochs=50, warmup_epochs=10)
        print(f"  Epoch {epoch:2d}: KL weight = {kl_weight:.2f}")


def main():
    """Run complete quickstart."""
    print("\n" + "🚀 " * 20)
    print("DIABETES VAE - QUICKSTART GUIDE")
    print("🚀 " * 20)
    
    # Run demonstrations
    quickstart_basic_setup()
    
    train_loader, val_loader = quickstart_data()
    
    device = get_device()
    
    baseline_model = quickstart_baseline_vae(device)
    
    multimodal_model = quickstart_multimodal_vae(device)
    
    quickstart_training_loss()
    
    # Next steps
    print_section("NEXT STEPS")
    print("To train models on your own data:")
    print("  1. Prepare your CGM, insulin, and physiology data in the format shown above")
    print("  2. Modify src/dataset.py to load your data (OhioT1DMDataset or MetaboNetDataset)")
    print("  3. Run: python scripts/train_baseline_vae.py")
    print("  4. Run: python scripts/train_multimodal_vae.py")
    print("  5. Evaluate: python scripts/evaluate_models.py")
    
    print_section("KEY FEATURES TO EXPLORE")
    features = [
        "Product-of-Experts (PoE) fusion for combining modalities",
        "KL annealing to prevent posterior collapse",
        "Missing modality handling (graceful degradation)",
        "UMAP visualization of latent space",
        "Inter-patient similarity analysis (RQ2)",
        "Latent trajectory analysis for drift (RQ1)",
    ]
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")
    
    print("\n" + "✅ " * 20)
    print("Quickstart complete! Happy training!")
    print("✅ " * 20 + "\n")


if __name__ == '__main__':
    main()
