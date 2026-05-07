"""
Dataset classes for loading OhioT1DM and MetaboNet data.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class DiabeticPatientDayDataset(Dataset):
    """
    Dataset representing patient-day samples with multimodal data.
    
    Each sample consists of:
    - CGM signal: 288 glucose readings at 5-minute intervals (24 hours)
    - Insulin profile: bolus and basal insulin doses on same 5-minute grid
    - Physiology + activity: daily summaries of HR, steps, skin signals
    """
    
    def __init__(self,
                 cgm_data: np.ndarray,
                 insulin_data: np.ndarray,
                 physiology_data: Optional[np.ndarray] = None,
                 patient_ids: Optional[np.ndarray] = None,
                 dates: Optional[List[str]] = None,
                 clinical_targets: Optional[Dict[str, np.ndarray]] = None,
                 normalize: bool = True):
        """
        Args:
            cgm_data: (n_samples, 288) glucose readings
            insulin_data: (n_samples, 288) insulin doses
            physiology_data: (n_samples, n_physiology_features) or None
            patient_ids: (n_samples,) patient identifiers
            dates: list of date strings
            clinical_targets: dict with clinical metrics (time_in_range, glucose_var, etc.)
            normalize: whether to apply StandardScaler normalization
        """
        self.cgm_data = cgm_data.astype(np.float32)
        self.insulin_data = insulin_data.astype(np.float32)
        self.physiology_data = physiology_data.astype(np.float32) if physiology_data is not None else None
        self.patient_ids = patient_ids
        self.dates = dates if dates is not None else ["unknown"] * len(cgm_data)
        self.clinical_targets = clinical_targets or {}
        
        self.n_samples = len(cgm_data)
        self.modalities = ['cgm', 'insulin']
        if self.physiology_data is not None:
            self.modalities.append('physiology')
        
        # Normalization
        self.normalize = normalize
        self.scalers = {}
        if normalize:
            self._fit_scalers()
    
    def _fit_scalers(self):
        """Fit StandardScaler for each modality."""
        self.scalers['cgm'] = StandardScaler()
        self.scalers['cgm'].fit(self.cgm_data)
        
        self.scalers['insulin'] = StandardScaler()
        self.scalers['insulin'].fit(self.insulin_data)
        
        if self.physiology_data is not None:
            self.scalers['physiology'] = StandardScaler()
            self.scalers['physiology'].fit(self.physiology_data)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with:
        - cgm: (288,) tensor
        - insulin: (288,) tensor
        - physiology: (n_features,) tensor or None
        - patient_id: int or str
        - date: str
        - clinical_targets: dict with clinical metrics for this sample
        """
        item = {}
        
        # CGM
        cgm = self.cgm_data[idx].copy()
        if self.normalize:
            cgm = self.scalers['cgm'].transform(cgm.reshape(-1, 1)).flatten()
        item['cgm'] = torch.from_numpy(cgm.astype(np.float32))
        
        # Insulin
        insulin = self.insulin_data[idx].copy()
        if self.normalize:
            insulin = self.scalers['insulin'].transform(insulin.reshape(-1, 1)).flatten()
        item['insulin'] = torch.from_numpy(insulin.astype(np.float32))
        
        # Physiology (optional)
        if self.physiology_data is not None:
            phys = self.physiology_data[idx].copy()
            if self.normalize:
                phys = self.scalers['physiology'].transform(phys.reshape(1, -1)).flatten()
            item['physiology'] = torch.from_numpy(phys.astype(np.float32))
        else:
            item['physiology'] = None
        
        # Metadata
        item['patient_id'] = self.patient_ids[idx] if self.patient_ids is not None else idx
        item['date'] = self.dates[idx]
        
        # Clinical targets
        for key, values in self.clinical_targets.items():
            item[key] = values[idx] if isinstance(values, np.ndarray) else values
        
        return item


class OhioT1DMDataset(Dataset):
    """
    Placeholder for OhioT1DM dataset loading.
    Requires Data Use Agreement from Ohio University.
    """
    
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: path to OhioT1DM data directory
            split: "train", "val", or "test"
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"OhioT1DM data directory not found: {self.data_dir}\n"
                "Please download from Ohio University (requires DUA)"
            )
        
        # TODO: Implement actual data loading
        # Expected structure:
        # data_dir/
        #   patient_1/
        #     cgm.csv
        #     insulin.csv
        #     heart_rate.csv
        #     ...
        self._load_data()
    
    def _load_data(self):
        """Load OhioT1DM data from disk."""
        raise NotImplementedError("OhioT1DM data loading not yet implemented")
    
    def __len__(self) -> int:
        raise NotImplementedError()
    
    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError()


class MetaboNetDataset(Dataset):
    """
    Placeholder for MetaboNet dataset loading.
    Available at: metabo-net.org
    """
    
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: path to MetaboNet data directory
            split: "train", "val", or "test"
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"MetaboNet data directory not found: {self.data_dir}\n"
                "Please download from metabo-net.org"
            )
        
        # TODO: Implement actual data loading
        self._load_data()
    
    def _load_data(self):
        """Load MetaboNet data from disk."""
        raise NotImplementedError("MetaboNet data loading not yet implemented")
    
    def __len__(self) -> int:
        raise NotImplementedError()
    
    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError()


def create_dummy_dataset(n_samples: int = 100,
                         n_physiology_features: int = 5) -> DiabeticPatientDayDataset:
    """
    Create a dummy dataset for testing/development.
    
    Args:
        n_samples: number of patient-day samples
        n_physiology_features: number of physiology features
    
    Returns:
        DiabeticPatientDayDataset with synthetic data
    """
    cgm_data = np.random.normal(150, 50, (n_samples, 288))
    cgm_data = np.clip(cgm_data, 50, 400)  # Realistic glucose range
    
    insulin_data = np.random.exponential(2, (n_samples, 288))
    
    physiology_data = np.random.normal(0, 1, (n_samples, n_physiology_features))
    
    patient_ids = np.random.randint(1, 13, n_samples)
    dates = [f"2023-01-{(i % 30) + 1:02d}" for i in range(n_samples)]
    
    clinical_targets = {
        'time_in_range': np.random.uniform(0.5, 0.9, n_samples),
        'glucose_variability': np.random.uniform(15, 40, n_samples),
        'mean_glucose': np.random.normal(160, 30, n_samples),
    }
    
    return DiabeticPatientDayDataset(
        cgm_data=cgm_data,
        insulin_data=insulin_data,
        physiology_data=physiology_data,
        patient_ids=patient_ids,
        dates=dates,
        clinical_targets=clinical_targets,
        normalize=True
    )
