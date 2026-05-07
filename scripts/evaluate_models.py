"""
Evaluate trained models and compute metrics.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from torch.utils.data import DataLoader, random_split
from config import Config
from dataset import create_dummy_dataset
from models import BaselineVAE, MultimodalVAEWithPoE
from evaluation import LatentSpaceEvaluator
from utils import set_seed, get_device, load_model


def main():
    set_seed(42)
    device = get_device()
    config = Config()
    
    # Create dummy dataset
    print("Loading dataset...")
    dataset = create_dummy_dataset(n_samples=200, n_physiology_features=5)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    test_loader = DataLoader(test_set, batch_size=config.training.batch_size, shuffle=False)
    
    print(f"Test set: {len(test_set)} samples")
    
    # Create evaluator
    evaluator = LatentSpaceEvaluator(output_dir='results')
    
    # Example: Evaluate baseline VAE
    print("\n" + "="*50)
    print("Evaluating Baseline VAE")
    print("="*50)
    
    model_baseline = BaselineVAE(
        cgm_length=config.data.cgm_readings_per_day,
        latent_dim=config.model.latent_dim
    )
    model_baseline.to(device)
    
    # Get latent representations
    latent_codes, metadata = evaluator.get_latent_representations(
        model=model_baseline,
        dataloader=test_loader,
        device=device
    )
    
    print(f"Latent codes shape: {latent_codes.shape}")
    
    # Visualize
    try:
        evaluator.visualize_latent_space_umap(latent_codes, metadata)
    except Exception as e:
        print(f"Could not visualize UMAP: {e}")
    
    # Compute reconstruction quality
    recon_metrics = evaluator.compute_reconstruction_quality(
        model=model_baseline,
        dataloader=test_loader,
        device=device
    )
    print(f"\nReconstruction metrics: {recon_metrics}")
    
    # Compute inter-patient similarity
    similarity_metrics = evaluator.compute_inter_patient_similarity(latent_codes, metadata)
    print(f"Inter-patient similarity metrics: {similarity_metrics}")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
