"""
Evaluation metrics and visualization for metabolic state representations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
import pandas as pd

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class LatentSpaceEvaluator:
    """Evaluate quality of learned latent representations."""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_latent_representations(self,
                                  model: torch.nn.Module,
                                  dataloader,
                                  device: str = 'cpu') -> Tuple[np.ndarray, Dict]:
        """
        Extract latent representations for all samples.
        
        Returns:
            latent_codes: (n_samples, latent_dim)
            metadata: dict with sample metadata
        """
        model.eval()
        latent_codes = []
        metadata = {
            'patient_ids': [],
            'dates': [],
            'time_in_range': [],
            'glucose_variability': [],
            'mean_glucose': [],
        }
        
        with torch.no_grad():
            for batch in dataloader:
                # Prepare batch
                cgm = batch['cgm'].to(device) if batch.get('cgm') is not None else None
                insulin = batch['insulin'].to(device) if batch.get('insulin') is not None else None
                physiology = batch['physiology'].to(device) if batch.get('physiology') is not None else None
                
                # Encode
                if hasattr(model, 'encode'):
                    if hasattr(model, 'cgm_encoder'):  # Multimodal
                        z = model.encode(cgm=cgm, insulin=insulin, physiology=physiology)
                    else:  # Unimodal
                        z = model.encode(cgm)
                else:
                    # Manual encoding
                    if hasattr(model, 'cgm_encoder'):
                        outputs = model(cgm=cgm, insulin=insulin, physiology=physiology)
                        z = outputs['z']
                    else:
                        _, _, logvar = model(cgm)
                        z = logvar  # Fallback
                
                latent_codes.append(z.cpu().numpy())
                
                # Collect metadata
                metadata['patient_ids'].extend(batch['patient_id'].numpy())
                metadata['dates'].extend(batch['date'])
                
                for key in ['time_in_range', 'glucose_variability', 'mean_glucose']:
                    if key in batch:
                        metadata[key].extend(batch[key].numpy())
        
        latent_codes = np.vstack(latent_codes)
        
        # Convert lists to arrays
        for key in metadata:
            if isinstance(metadata[key], list) and len(metadata[key]) > 0:
                if isinstance(metadata[key][0], (int, float)):
                    metadata[key] = np.array(metadata[key])
        
        return latent_codes, metadata
    
    def visualize_latent_space_umap(self,
                                   latent_codes: np.ndarray,
                                   metadata: Dict,
                                   metric: str = 'time_in_range',
                                   n_neighbors: int = 15,
                                   min_dist: float = 0.1) -> None:
        """
        Visualize latent space using UMAP colored by clinical metric.
        """
        if not HAS_UMAP:
            print("UMAP not installed. Install with: pip install umap-learn")
            return
        
        if latent_codes.shape[1] > 2:
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
            embedding = reducer.fit_transform(latent_codes)
        else:
            embedding = latent_codes[:, :2]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Color by time in range
        if 'time_in_range' in metadata and len(metadata['time_in_range']) > 0:
            scatter = axes[0].scatter(embedding[:, 0], embedding[:, 1],
                                     c=metadata['time_in_range'],
                                     cmap='RdYlGn', s=30, alpha=0.6)
            axes[0].set_title('Latent Space colored by Time-In-Range')
            plt.colorbar(scatter, ax=axes[0])
        
        # Color by glucose variability
        if 'glucose_variability' in metadata and len(metadata['glucose_variability']) > 0:
            scatter = axes[1].scatter(embedding[:, 0], embedding[:, 1],
                                     c=metadata['glucose_variability'],
                                     cmap='coolwarm', s=30, alpha=0.6)
            axes[1].set_title('Latent Space colored by Glucose Variability')
            plt.colorbar(scatter, ax=axes[1])
        
        # Color by patient ID
        if 'patient_ids' in metadata and len(metadata['patient_ids']) > 0:
            scatter = axes[2].scatter(embedding[:, 0], embedding[:, 1],
                                     c=metadata['patient_ids'],
                                     cmap='tab20', s=30, alpha=0.6)
            axes[2].set_title('Latent Space colored by Patient ID')
            plt.colorbar(scatter, ax=axes[2])
        
        for ax in axes:
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
        
        plt.tight_layout()
        output_path = self.output_dir / 'latent_space_umap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved UMAP visualization: {output_path}")
        plt.close()
    
    def compute_inter_patient_similarity(self,
                                        latent_codes: np.ndarray,
                                        metadata: Dict) -> Dict:
        """
        Compute correlation between latent space distances and clinical similarities.
        
        Research Question 2: Does distance in latent space reflect clinical similarity?
        """
        if 'patient_ids' not in metadata or len(metadata['patient_ids']) == 0:
            return {}
        
        # Group by patient
        unique_patients = np.unique(metadata['patient_ids'])
        patient_latents = {}
        patient_tir = {}
        patient_gluc_var = {}
        
        for pid in unique_patients:
            mask = metadata['patient_ids'] == pid
            patient_latents[pid] = latent_codes[mask].mean(axis=0)
            
            if 'time_in_range' in metadata and len(metadata['time_in_range']) > 0:
                patient_tir[pid] = metadata['time_in_range'][mask].mean()
            
            if 'glucose_variability' in metadata and len(metadata['glucose_variability']) > 0:
                patient_gluc_var[pid] = metadata['glucose_variability'][mask].mean()
        
        # Compute pairwise distances and clinical differences
        latent_dists = squareform(pdist([patient_latents[p] for p in unique_patients]))
        
        results = {}
        
        if patient_tir:
            tir_diffs = squareform(pdist([[patient_tir[p]] for p in unique_patients]))
            corr, pval = spearmanr(latent_dists[np.triu_indices_from(latent_dists, k=1)],
                                  tir_diffs[np.triu_indices_from(tir_diffs, k=1)])
            results['tir_correlation'] = {'corr': corr, 'pval': pval}
        
        if patient_gluc_var:
            var_diffs = squareform(pdist([[patient_gluc_var[p]] for p in unique_patients]))
            corr, pval = spearmanr(latent_dists[np.triu_indices_from(latent_dists, k=1)],
                                  var_diffs[np.triu_indices_from(var_diffs, k=1)])
            results['glucose_var_correlation'] = {'corr': corr, 'pval': pval}
        
        return results
    
    def test_missing_modality_robustness(self,
                                        model: torch.nn.Module,
                                        dataloader,
                                        device: str = 'cpu',
                                        modality_combos: Optional[List[str]] = None) -> Dict:
        """
        Test model robustness by masking different modalities at test time.
        
        Returns:
            dict with reconstruction errors and latent stability metrics
        """
        if not hasattr(model, 'encode'):
            return {'error': 'Model does not support testing missing modalities'}
        
        if modality_combos is None:
            modality_combos = [
                'cgm+insulin+physiology',
                'cgm+insulin',
                'cgm+physiology',
                'insulin+physiology',
                'cgm_only',
                'insulin_only',
                'physiology_only',
            ]
        
        results = {combo: {'recon_errors': {}, 'latent_stability': 0} for combo in modality_combos}
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                cgm = batch['cgm'].to(device)
                insulin = batch['insulin'].to(device)
                physiology = batch['physiology'].to(device) if batch.get('physiology') is not None else None
                
                # Full encoding (reference)
                z_full = model.encode(cgm=cgm, insulin=insulin, physiology=physiology)
                
                # Test different modality combinations
                for combo in modality_combos:
                    if combo == 'cgm+insulin+physiology':
                        z = z_full
                    elif combo == 'cgm+insulin':
                        z = model.encode(cgm=cgm, insulin=insulin, physiology=None)
                    elif combo == 'cgm+physiology':
                        z = model.encode(cgm=cgm, insulin=None, physiology=physiology)
                    elif combo == 'insulin+physiology':
                        z = model.encode(cgm=None, insulin=insulin, physiology=physiology)
                    elif combo == 'cgm_only':
                        z = model.encode(cgm=cgm, insulin=None, physiology=None)
                    elif combo == 'insulin_only':
                        z = model.encode(cgm=None, insulin=insulin, physiology=None)
                    elif combo == 'physiology_only':
                        z = model.encode(cgm=None, insulin=None, physiology=physiology)
                    
                    # Latent stability (distance from full)
                    latent_stability = torch.norm(z - z_full, dim=1).mean().item()
                    
                    # TODO: Compute reconstruction errors if decoders available
        
        return results
    
    def compute_reconstruction_quality(self,
                                      model: torch.nn.Module,
                                      dataloader,
                                      device: str = 'cpu') -> Dict[str, float]:
        """Compute reconstruction error metrics."""
        model.eval()
        
        recon_errors = {'cgm': [], 'insulin': [], 'physiology': []}
        
        with torch.no_grad():
            for batch in dataloader:
                cgm = batch['cgm'].to(device) if batch.get('cgm') is not None else None
                insulin = batch['insulin'].to(device) if batch.get('insulin') is not None else None
                physiology = batch['physiology'].to(device) if batch.get('physiology') is not None else None
                
                if hasattr(model, 'cgm_encoder'):  # Multimodal
                    outputs = model(cgm=cgm, insulin=insulin, physiology=physiology)
                    
                    if 'cgm_recon' in outputs and cgm is not None:
                        mse = torch.mean((outputs['cgm_recon'] - cgm) ** 2).item()
                        recon_errors['cgm'].append(mse)
                    
                    if 'insulin_recon' in outputs and insulin is not None:
                        mse = torch.mean((outputs['insulin_recon'] - insulin) ** 2).item()
                        recon_errors['insulin'].append(mse)
                    
                    if 'physiology_recon' in outputs and physiology is not None:
                        mse = torch.mean((outputs['physiology_recon'] - physiology) ** 2).item()
                        recon_errors['physiology'].append(mse)
                else:  # Unimodal
                    recon, _, _ = model(cgm)
                    mse = torch.mean((recon - cgm) ** 2).item()
                    recon_errors['cgm'].append(mse)
        
        # Average errors
        results = {}
        for modality, errors in recon_errors.items():
            if errors:
                results[f'{modality}_recon_mse'] = np.mean(errors)
                results[f'{modality}_recon_std'] = np.std(errors)
        
        return results
