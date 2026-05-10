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
    MetaboNet dataset: patient-day samples from preprocessed numpy arrays.

    Run scripts/preprocess_metabonet.py once to build data/processed/{split}/ from
    the raw train.parquet / test.parquet files.

    Each sample:
      cgm        (288,)  glucose mg/dL at 5-min intervals, linearly interpolated
      insulin    (288,)  total insulin units at 5-min intervals
      physiology (5,)    daily summaries: HR_mean, steps_sum, GSR_mean,
                         skin_temp_mean, calories_sum
    """

    PHYSIOLOGY_FEATURES = [
        'heartrate', 'steps', 'galvanic_skin_response', 'skin_temp', 'calories_burned'
    ]

    def __init__(self, processed_dir: str, split: str = "train", normalize: bool = True):
        """
        Args:
            processed_dir: path to data/processed/
            split: "train" or "test"
            normalize: apply StandardScaler per modality
        """
        self.split_dir = Path(processed_dir) / split
        self.split = split
        self.normalize = normalize

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Processed data not found: {self.split_dir}\n"
                "Run: python scripts/preprocess_metabonet.py"
            )

        self._load_data()

        self.modalities = ['cgm', 'insulin', 'physiology']
        self.scalers: Dict = {}
        if normalize:
            self._fit_scalers()

    def _load_data(self):
        self.cgm_data       = np.load(self.split_dir / 'cgm.npy',        mmap_mode='r')
        self.insulin_data   = np.load(self.split_dir / 'insulin.npy',    mmap_mode='r')
        self.physiology_data = np.load(self.split_dir / 'physiology.npy', mmap_mode='r')
        self.metadata       = pd.read_parquet(self.split_dir / 'metadata.parquet')
        self.n_samples      = len(self.cgm_data)

    def _fit_scalers(self):
        # Copy small slice into RAM for fitting scalers; mmap stays on disk
        cgm_fit = np.array(self.cgm_data, dtype=np.float32)
        ins_fit = np.array(self.insulin_data, dtype=np.float32)
        phy_fit = np.array(self.physiology_data, dtype=np.float32)

        # Physiology has NaN for missing wearable days — fit on observed only
        phy_obs = phy_fit[~np.isnan(phy_fit).any(axis=1)]

        self.scalers['cgm'] = StandardScaler().fit(cgm_fit)
        self.scalers['insulin'] = StandardScaler().fit(ins_fit)
        if len(phy_obs):
            self.scalers['physiology'] = StandardScaler().fit(phy_obs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict:
        cgm = np.array(self.cgm_data[idx], dtype=np.float32)
        if self.normalize and 'cgm' in self.scalers:
            cgm = self.scalers['cgm'].transform(cgm.reshape(-1, 1)).flatten()

        insulin = np.array(self.insulin_data[idx], dtype=np.float32)
        if self.normalize and 'insulin' in self.scalers:
            insulin = self.scalers['insulin'].transform(insulin.reshape(-1, 1)).flatten()

        phys = np.array(self.physiology_data[idx], dtype=np.float32)
        if self.normalize and 'physiology' in self.scalers:
            if not np.isnan(phys).any():
                phys = self.scalers['physiology'].transform(phys.reshape(1, -1)).flatten()
            else:
                phys = np.zeros_like(phys)  # missing wearable day → zero vector

        row = self.metadata.iloc[idx]
        return {
            'cgm':        torch.from_numpy(cgm),
            'insulin':    torch.from_numpy(insulin),
            'physiology': torch.from_numpy(phys),
            'patient_id': row['patient_id'],
            'date':       row['date'],
        }


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
