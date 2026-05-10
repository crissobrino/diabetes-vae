"""
Preprocess MetaboNet parquet files into per-patient-day numpy arrays.

Reads train.parquet and test.parquet from data/, outputs:
  data/processed/{split}/
    cgm.npy          (N, 288) float32  -- glucose mg/dL, NaN-interpolated
    insulin.npy      (N, 288) float32  -- total insulin units
    physiology.npy   (N, 5)  float32  -- daily summary: HR, steps, GSR, skin_temp, calories
    metadata.parquet          -- patient_id, date, source_file, n_cgm_readings, gender, age

Usage:
    python scripts/preprocess_metabonet.py
    python scripts/preprocess_metabonet.py --min-cgm-fraction 0.4 --batch-size 500000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Physiology features extracted as daily summaries (mean, except steps/calories = sum)
PHYSIOLOGY_COLS = ['heartrate', 'steps', 'galvanic_skin_response', 'skin_temp', 'calories_burned']
PHYSIOLOGY_AGG = {'heartrate': 'mean', 'steps': 'sum', 'galvanic_skin_response': 'mean',
                  'skin_temp': 'mean', 'calories_burned': 'sum'}

SLOTS_PER_DAY = 288  # 24h * 60min / 5min


def slot_index(timestamps: pd.Series) -> pd.Series:
    """Convert timestamps to 5-minute slot indices (0–287)."""
    return (timestamps.dt.hour * 60 + timestamps.dt.minute) // 5


def interpolate_cgm(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN gaps in a 288-element CGM array."""
    s = pd.Series(arr)
    s = s.interpolate(method='linear', limit=12)  # max 1-hour gap
    s = s.ffill().bfill()                         # fill edges
    return s.values.astype(np.float32)


def process_parquet(path: Path, min_cgm_fraction: float, batch_size: int):
    """
    Stream through a parquet file and aggregate into patient-day arrays.
    Returns (cgm, insulin, physiology, metadata) as lists.
    """
    # patient_day_key -> dict with accumulation arrays
    days: dict = {}

    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    print(f"  {path.name}: {total_rows:,} rows, {pf.metadata.num_row_groups} row groups")

    for batch in tqdm(pf.iter_batches(batch_size=batch_size), desc=f"  {path.stem}",
                      total=total_rows // batch_size + 1):
        df = batch.to_pandas()

        df['_day'] = df['date'].dt.date
        df['_slot'] = slot_index(df['date']).clip(0, SLOTS_PER_DAY - 1)

        for (pid, day), grp in df.groupby(['id', '_day'], sort=False):
            key = (pid, day)
            if key not in days:
                # Take static patient metadata from first occurrence
                row0 = grp.iloc[0]
                days[key] = {
                    'cgm':          np.full(SLOTS_PER_DAY, np.nan, dtype=np.float32),
                    'insulin':      np.zeros(SLOTS_PER_DAY, dtype=np.float32),
                    'phys_sum':     np.zeros(len(PHYSIOLOGY_COLS), dtype=np.float64),
                    'phys_count':   np.zeros(len(PHYSIOLOGY_COLS), dtype=np.int32),
                    'n_cgm':        0,
                    'patient_id':   pid,
                    'date':         str(day),
                    'source_file':  row0.get('source_file', ''),
                    'gender':       row0.get('gender', None),
                    'age':          row0.get('age', np.nan),
                }

            entry = days[key]
            slots = grp['_slot'].values.astype(int)

            # CGM (sparse — only present in some rows)
            cgm_vals = grp['CGM'].values.astype(np.float32)
            cgm_ok = ~np.isnan(cgm_vals)
            if cgm_ok.any():
                entry['cgm'][slots[cgm_ok]] = cgm_vals[cgm_ok]
                entry['n_cgm'] += int(cgm_ok.sum())

            # Insulin (total; 0 when not explicitly given)
            ins_vals = grp['insulin'].values
            ins_ok = ~np.isnan(ins_vals)
            if ins_ok.any():
                entry['insulin'][slots[ins_ok]] = ins_vals[ins_ok].astype(np.float32)

            # Physiology running mean/sum
            for j, col in enumerate(PHYSIOLOGY_COLS):
                if col in grp.columns:
                    vals = grp[col].dropna().values
                    if len(vals):
                        entry['phys_sum'][j] += vals.sum()
                        entry['phys_count'][j] += len(vals)

    # Filter and convert
    min_cgm = int(min_cgm_fraction * SLOTS_PER_DAY)
    kept = [v for v in days.values() if v['n_cgm'] >= min_cgm]
    print(f"  Patient-days before filter: {len(days):,}  after (≥{min_cgm} CGM readings): {len(kept):,}")

    if not kept:
        raise RuntimeError("No patient-days passed the CGM coverage filter.")

    cgm_arr = np.stack([interpolate_cgm(e['cgm']) for e in kept])
    insulin_arr = np.stack([e['insulin'] for e in kept])

    phys_arr = np.zeros((len(kept), len(PHYSIOLOGY_COLS)), dtype=np.float32)
    for i, e in enumerate(kept):
        with np.errstate(invalid='ignore'):
            mask = e['phys_count'] > 0
            phys_arr[i, mask] = (e['phys_sum'][mask] / e['phys_count'][mask]).astype(np.float32)
        phys_arr[i, ~mask] = np.nan

    meta = pd.DataFrame({
        'patient_id':    [e['patient_id']   for e in kept],
        'date':          [e['date']          for e in kept],
        'source_file':   [e['source_file']   for e in kept],
        'n_cgm_readings':[e['n_cgm']         for e in kept],
        'gender':        [e['gender']        for e in kept],
        'age':           [e['age']           for e in kept],
    })

    return cgm_arr, insulin_arr, phys_arr, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', type=Path)
    parser.add_argument('--output-dir', default='data/processed', type=Path)
    parser.add_argument('--min-cgm-fraction', default=0.5, type=float,
                        help='Minimum fraction of day that must have CGM readings (default 0.5)')
    parser.add_argument('--batch-size', default=1_000_000, type=int,
                        help='Rows per batch during streaming (default 1M)')
    args = parser.parse_args()

    splits = {'train': args.data_dir / 'train.parquet',
              'test':  args.data_dir / 'test.parquet'}

    for split, path in splits.items():
        if not path.exists():
            print(f"Skipping {split}: {path} not found")
            continue

        print(f"\nProcessing {split}...")
        out_dir = args.output_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        cgm, insulin, physiology, meta = process_parquet(path, args.min_cgm_fraction, args.batch_size)

        np.save(out_dir / 'cgm.npy',        cgm)
        np.save(out_dir / 'insulin.npy',     insulin)
        np.save(out_dir / 'physiology.npy',  physiology)
        meta.to_parquet(out_dir / 'metadata.parquet', index=False)

        print(f"  Saved to {out_dir}/")
        print(f"  cgm.npy:       {cgm.shape}")
        print(f"  insulin.npy:   {insulin.shape}")
        print(f"  physiology.npy:{physiology.shape}")
        print(f"  metadata:      {len(meta)} rows")

    print("\nDone.")


if __name__ == '__main__':
    sys.exit(main())
