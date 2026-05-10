import numpy as np, pandas as pd
cgm  = np.load('data/processed/train/cgm.npy', mmap_mode='r')
meta = pd.read_parquet('data/processed/train/metadata.parquet')

print(cgm.shape)                          # (N, 288)
print(meta['patient_id'].nunique())       # unique patients
print(meta['source_file'].value_counts()) # OhioT1DM vs AZT1D mix
print(np.isnan(cgm).mean())              # should be ~0 after interpolation
