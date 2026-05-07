"""
Training pipeline for VAE models with KL annealing and monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from tqdm import tqdm
import json
from datetime import datetime


class VAETrainer:
    """Trainer class for VAE models."""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), 
                                   lr=learning_rate,
                                   weight_decay=weight_decay)
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': [],
        }
    
    def vae_loss(self,
                 recon: torch.Tensor,
                 target: torch.Tensor,
                 mu: torch.Tensor,
                 logvar: torch.Tensor,
                 kl_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss: ELBO = reconstruction loss + KL divergence.
        
        Args:
            recon: reconstructed data
            target: original data
            mu: latent mean
            logvar: latent log-variance
            kl_weight: weight for KL term (for annealing)
        
        Returns:
            total_loss, reconstruction_loss, kl_loss
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss(reduction='mean')(recon, target)
        
        # KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def multimodal_vae_loss(self,
                           outputs: Dict,
                           targets: Dict,
                           kl_weight: float = 1.0,
                           modality_weights: Dict[str, float] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multimodal VAE loss across all available modalities.
        
        Args:
            outputs: dict with 'mu', 'logvar', and '*_recon' keys
            targets: dict with modality targets
            kl_weight: weight for KL term
            modality_weights: weight for each modality reconstruction
        
        Returns:
            total_loss, loss_breakdown dict
        """
        if modality_weights is None:
            modality_weights = {'cgm': 1.0, 'insulin': 1.0, 'physiology': 1.0}
        
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Reconstruction losses for available modalities
        recon_loss = torch.tensor(0.0, device=mu.device)
        losses = {'kl': kl_loss.item()}
        
        if 'cgm_recon' in outputs and 'cgm' in targets:
            cgm_loss = nn.MSELoss()(outputs['cgm_recon'], targets['cgm'])
            recon_loss = recon_loss + modality_weights['cgm'] * cgm_loss
            losses['cgm_recon'] = cgm_loss.item()
        
        if 'insulin_recon' in outputs and 'insulin' in targets:
            insulin_loss = nn.MSELoss()(outputs['insulin_recon'], targets['insulin'])
            recon_loss = recon_loss + modality_weights['insulin'] * insulin_loss
            losses['insulin_recon'] = insulin_loss.item()
        
        if 'physiology_recon' in outputs and 'physiology' in targets:
            phys_loss = nn.MSELoss()(outputs['physiology_recon'], targets['physiology'])
            recon_loss = recon_loss + modality_weights['physiology'] * phys_loss
            losses['physiology_recon'] = phys_loss.item()
        
        total_loss = recon_loss + kl_weight * kl_loss
        losses['total'] = total_loss.item()
        losses['recon_total'] = recon_loss.item()
        
        return total_loss, losses
    
    def get_kl_weight(self, epoch: int, total_epochs: int, warmup_epochs: int = 10) -> float:
        """
        KL annealing schedule: linearly increase weight from 0 to 1 over warmup_epochs.
        """
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   epoch: int,
                   total_epochs: int,
                   kl_annealing: bool = True,
                   warmup_epochs: int = 10,
                   modality_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            dict with loss metrics
        """
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        
        kl_weight = self.get_kl_weight(epoch, total_epochs, warmup_epochs) if kl_annealing else 1.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Prepare batch data
            cgm = batch['cgm'].to(self.device) if batch.get('cgm') is not None else None
            insulin = batch['insulin'].to(self.device) if batch.get('insulin') is not None else None
            physiology = batch['physiology'].to(self.device) if batch.get('physiology') is not None else None
            
            # Check if this is multimodal or unimodal model
            if hasattr(self.model, 'cgm_encoder') and hasattr(self.model, 'insulin_encoder'):
                # Multimodal model
                outputs = self.model(cgm=cgm, insulin=insulin, physiology=physiology)
                targets = {'cgm': cgm, 'insulin': insulin, 'physiology': physiology}
                targets = {k: v for k, v in targets.items() if v is not None}
                
                loss, loss_dict = self.multimodal_vae_loss(outputs, targets, kl_weight, modality_weights)
                total_recon += loss_dict.get('recon_total', 0)
                total_kl += loss_dict.get('kl', 0)
            else:
                # Baseline unimodal model
                recon, mu, logvar = self.model(cgm)
                loss, recon_loss, kl_loss = self.vae_loss(recon, cgm, mu, logvar, kl_weight)
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item(),
                'kl_w': kl_weight
            })
        
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_recon'].append(avg_recon)
        self.history['train_kl'].append(avg_kl)
        
        return {
            'loss': avg_loss,
            'recon': avg_recon,
            'kl': avg_kl,
            'kl_weight': kl_weight
        }
    
    @torch.no_grad()
    def validate(self,
                val_loader: DataLoader,
                kl_weight: float = 1.0,
                modality_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Validate on validation set.
        """
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        
        for batch in val_loader:
            cgm = batch['cgm'].to(self.device) if batch.get('cgm') is not None else None
            insulin = batch['insulin'].to(self.device) if batch.get('insulin') is not None else None
            physiology = batch['physiology'].to(self.device) if batch.get('physiology') is not None else None
            
            if hasattr(self.model, 'cgm_encoder') and hasattr(self.model, 'insulin_encoder'):
                outputs = self.model(cgm=cgm, insulin=insulin, physiology=physiology)
                targets = {'cgm': cgm, 'insulin': insulin, 'physiology': physiology}
                targets = {k: v for k, v in targets.items() if v is not None}
                
                loss, loss_dict = self.multimodal_vae_loss(outputs, targets, kl_weight, modality_weights)
                total_recon += loss_dict.get('recon_total', 0)
                total_kl += loss_dict.get('kl', 0)
            else:
                recon, mu, logvar = self.model(cgm)
                loss, recon_loss, kl_loss = self.vae_loss(recon, cgm, mu, logvar, kl_weight)
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_recon'].append(avg_recon)
        self.history['val_kl'].append(avg_kl)
        
        return {
            'loss': avg_loss,
            'recon': avg_recon,
            'kl': avg_kl,
        }
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            kl_annealing: bool = True,
            warmup_epochs: int = 10,
            save_dir: str = 'models',
            save_freq: int = 5,
            modality_weights: Optional[Dict[str, float]] = None):
        """
        Full training loop.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(
                train_loader, epoch, epochs,
                kl_annealing=kl_annealing,
                warmup_epochs=warmup_epochs,
                modality_weights=modality_weights
            )
            
            # Validate
            kl_weight = self.get_kl_weight(epoch, epochs, warmup_epochs) if kl_annealing else 1.0
            val_metrics = self.validate(val_loader, kl_weight, modality_weights)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Recon: {val_metrics['recon']:.4f}, KL: {val_metrics['kl']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'history': self.history,
                    'metrics': {'train': train_metrics, 'val': val_metrics}
                }
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_path = save_dir / "best_model.pt"
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  New best model saved: {best_model_path}")
    
    def save_history(self, path: str):
        """Save training history to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
