"""
Kaggle GPU training script — Diabetes Multimodal VAE
=====================================================

SETUP (do this once before running):
--------------------------------------
1. Upload your parquet files as a Kaggle Dataset:
      Kaggle > Datasets > New Dataset
      Upload train.parquet and test.parquet
      Name it e.g. "metabonet-diabetes"

2. In your Kaggle Notebook:
      Add the dataset above via "+ Add Data"
      Enable GPU:  Settings > Accelerator > GPU T4 x2  (or P100)
      Set Persistence: Files only

3. Upload this script (and the rest of the repo) as a second dataset,
   OR paste this script directly into the notebook cell.

Paths assumed:
   /kaggle/input/metabonet-diabetes/train.parquet
   /kaggle/input/metabonet-diabetes/test.parquet
   /kaggle/working/                  ← all outputs saved here
"""

# ── 0. Install missing packages ───────────────────────────────────────────────
import subprocess, sys

def pip(pkg):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)

pip('pyarrow')
pip('umap-learn')

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 2. Paths & config ────────────────────────────────────────────────────────
ON_KAGGLE = Path('/kaggle').exists()

RAW_DIR   = Path('/kaggle/input/metabonet-diabetes') if ON_KAGGLE else Path('data')
WORK_DIR  = Path('/kaggle/working')                  if ON_KAGGLE else Path('kaggle_output')
PROC_DIR  = WORK_DIR / 'processed'
MODEL_DIR = WORK_DIR / 'models'
WORK_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS  = 4 if ON_KAGGLE else 0   # Windows needs 0; Linux Kaggle is fine with 4
BATCH_SIZE   = 128                      # larger batch for GPU
NUM_EPOCHS   = 100
LR           = 3e-4
KL_WARMUP    = 25
LATENT_DIM   = 8
ENC_CHANNELS = [16, 32, 64]
DEC_CHANNELS = [64, 32, 16]
PHYSIO_DIM   = 5
MIN_CGM_FRAC = 0.5
CHECKPOINT_FREQ = 5

print(f"Device : {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"Kaggle : {ON_KAGGLE}")


# ── 3. Preprocessing ──────────────────────────────────────────────────────────
PHYSIO_COLS = ['heartrate', 'steps', 'galvanic_skin_response',
               'skin_temp', 'calories_burned']
SLOTS = 288


def slot_index(ts: pd.Series) -> pd.Series:
    return (ts.dt.hour * 60 + ts.dt.minute) // 5


def interpolate_cgm(arr: np.ndarray) -> np.ndarray:
    s = pd.Series(arr).interpolate(method='linear', limit=12).ffill().bfill()
    return s.values.astype(np.float32)


def preprocess_split(parquet_path: Path, out_dir: Path,
                     min_cgm_frac: float = MIN_CGM_FRAC,
                     batch_size: int = 1_000_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    days: dict = {}
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    print(f"  {parquet_path.name}: {total:,} rows")

    for batch in tqdm(pf.iter_batches(batch_size=batch_size),
                      total=total // batch_size + 1, desc='  batches'):
        df = batch.to_pandas()
        df['_day']  = df['date'].dt.date
        df['_slot'] = slot_index(df['date']).clip(0, SLOTS - 1)

        for (pid, day), grp in df.groupby(['id', '_day'], sort=False):
            key = (pid, day)
            if key not in days:
                r = grp.iloc[0]
                days[key] = {
                    'cgm':        np.full(SLOTS, np.nan, dtype=np.float32),
                    'insulin':    np.zeros(SLOTS, dtype=np.float32),
                    'phys_sum':   np.zeros(len(PHYSIO_COLS), dtype=np.float64),
                    'phys_count': np.zeros(len(PHYSIO_COLS), dtype=np.int32),
                    'n_cgm':      0,
                    'patient_id': pid,
                    'date':       str(day),
                    'source':     r.get('source_file', ''),
                    'gender':     r.get('gender', None),
                    'age':        r.get('age', np.nan),
                }
            e     = days[key]
            slots = grp['_slot'].values.astype(int)

            cgm_v = grp['CGM'].values.astype(np.float32)
            ok    = ~np.isnan(cgm_v)
            if ok.any():
                e['cgm'][slots[ok]] = cgm_v[ok]
                e['n_cgm'] += int(ok.sum())

            ins_v = grp['insulin'].values
            ok2   = ~np.isnan(ins_v)
            if ok2.any():
                e['insulin'][slots[ok2]] = ins_v[ok2].astype(np.float32)

            for j, col in enumerate(PHYSIO_COLS):
                if col in grp.columns:
                    vals = grp[col].dropna().values
                    if len(vals):
                        e['phys_sum'][j]   += vals.sum()
                        e['phys_count'][j] += len(vals)

    min_cgm = int(min_cgm_frac * SLOTS)
    kept = [v for v in days.values() if v['n_cgm'] >= min_cgm]
    print(f"  Patient-days: {len(days):,} total → {len(kept):,} after filter")

    cgm_arr  = np.stack([interpolate_cgm(e['cgm'])  for e in kept])
    ins_arr  = np.stack([e['insulin']                for e in kept])

    phys_arr = np.zeros((len(kept), len(PHYSIO_COLS)), dtype=np.float32)
    for i, e in enumerate(kept):
        mask = e['phys_count'] > 0
        with np.errstate(invalid='ignore'):
            phys_arr[i, mask] = (e['phys_sum'][mask] /
                                 e['phys_count'][mask]).astype(np.float32)
        phys_arr[i, ~mask] = np.nan

    meta = pd.DataFrame({k: [e[k] for e in kept]
                         for k in ('patient_id', 'date', 'source',
                                   'gender', 'age', 'n_cgm')})

    np.save(out_dir / 'cgm.npy',        cgm_arr)
    np.save(out_dir / 'insulin.npy',    ins_arr)
    np.save(out_dir / 'physiology.npy', phys_arr)
    meta.to_parquet(out_dir / 'metadata.parquet', index=False)
    print(f"  Saved to {out_dir}  cgm={cgm_arr.shape}")


def run_preprocessing():
    for split in ('train', 'test'):
        out = PROC_DIR / split
        if (out / 'cgm.npy').exists():
            print(f"Preprocessing for '{split}' already done, skipping.")
            continue
        src = RAW_DIR / f'{split}.parquet'
        if not src.exists():
            raise FileNotFoundError(f"Not found: {src}\n"
                                    "Add the dataset in Kaggle > +Add Data")
        print(f"\nPreprocessing {split}...")
        preprocess_split(src, out)


# ── 4. Dataset ────────────────────────────────────────────────────────────────
class MetaboNetDataset(Dataset):
    def __init__(self, proc_dir: Path, split: str, normalize: bool = True):
        self.split_dir = proc_dir / split
        self.cgm_data       = np.load(self.split_dir / 'cgm.npy',        mmap_mode='r')
        self.insulin_data   = np.load(self.split_dir / 'insulin.npy',    mmap_mode='r')
        self.physiology_data= np.load(self.split_dir / 'physiology.npy', mmap_mode='r')
        self.metadata       = pd.read_parquet(self.split_dir / 'metadata.parquet')
        self.n              = len(self.cgm_data)
        self.scalers        = {}
        if normalize:
            self._fit_scalers()

    def _fit_scalers(self):
        cgm = np.array(self.cgm_data,        dtype=np.float32)
        ins = np.array(self.insulin_data,     dtype=np.float32)
        phy = np.array(self.physiology_data,  dtype=np.float32)
        phy_obs = phy[~np.isnan(phy).any(axis=1)]
        self.scalers['cgm']     = StandardScaler().fit(cgm)
        self.scalers['insulin'] = StandardScaler().fit(ins)
        if len(phy_obs):
            self.scalers['physiology'] = StandardScaler().fit(phy_obs)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        cgm = np.array(self.cgm_data[idx],    dtype=np.float32)
        ins = np.array(self.insulin_data[idx], dtype=np.float32)
        phy = np.array(self.physiology_data[idx], dtype=np.float32)

        if 'cgm' in self.scalers:
            cgm = self.scalers['cgm'].transform(cgm.reshape(1, -1)).flatten()
        if 'insulin' in self.scalers:
            ins = self.scalers['insulin'].transform(ins.reshape(1, -1)).flatten()
        if np.isnan(phy).any():
            phy = np.zeros_like(phy)
        elif 'physiology' in self.scalers:
            phy = self.scalers['physiology'].transform(phy.reshape(1, -1)).flatten()

        row = self.metadata.iloc[idx]
        return {'cgm': torch.from_numpy(cgm), 'insulin': torch.from_numpy(ins),
                'physiology': torch.from_numpy(phy),
                'patient_id': row['patient_id'], 'date': row['date']}


# ── 5. Model ──────────────────────────────────────────────────────────────────
class ConvEncoder1D(nn.Module):
    def __init__(self, input_length, channels, latent_dim=8,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        layers, in_ch = [], 1
        for out_ch in channels:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding), nn.ReLU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc_mu     = nn.Linear(in_ch * input_length, latent_dim)
        self.fc_logvar = nn.Linear(in_ch * input_length, latent_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder1D(nn.Module):
    def __init__(self, output_length, channels, latent_dim=8,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.out_len = output_length
        self.start_ch = channels[0]
        # FC expands to full sequence length — symmetric with encoder
        self.fc = nn.Linear(latent_dim, channels[0] * output_length)
        layers, in_ch = [], channels[0]
        for out_ch in channels[1:] + [1]:
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride, padding))
            if out_ch != 1: layers.append(nn.ReLU())
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), self.start_ch, self.out_len)
        return self.deconv(x).squeeze(1)


class MultimodalVAE(nn.Module):
    def __init__(self, seq_len=288, physio_dim=5,
                 enc_ch=None, dec_ch=None, latent_dim=8):
        super().__init__()
        enc_ch = enc_ch or [16, 32, 64]
        dec_ch = dec_ch or [64, 32, 16]
        self.latent_dim = latent_dim

        self.cgm_enc    = ConvEncoder1D(seq_len,    enc_ch, latent_dim)
        self.insulin_enc= ConvEncoder1D(seq_len,    enc_ch, latent_dim)
        self.phys_enc   = nn.Sequential(nn.Linear(physio_dim, 64), nn.ReLU(),
                                        nn.Linear(64, 32), nn.ReLU())
        self.phys_mu    = nn.Linear(32, latent_dim)
        self.phys_logvar= nn.Linear(32, latent_dim)

        self.cgm_dec    = ConvDecoder1D(seq_len, dec_ch, latent_dim)
        self.insulin_dec= ConvDecoder1D(seq_len, dec_ch, latent_dim)
        self.phys_dec   = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(),
                                        nn.Linear(32, 64), nn.ReLU(),
                                        nn.Linear(64, physio_dim))

    def product_of_experts(self, mu_list, logvar_list):
        prec_sum = torch.zeros_like(mu_list[0])
        wmu_sum  = torch.zeros_like(mu_list[0])
        for mu, lv in zip(mu_list, logvar_list):
            prec = 1.0 / (lv.clamp(-10, 10).exp() + 1e-8)
            prec_sum += prec
            wmu_sum  += prec * mu
        mu_poe     = wmu_sum  / (prec_sum + 1e-8)
        logvar_poe = torch.log(1.0 / (prec_sum + 1e-8) + 1e-8)
        return mu_poe, logvar_poe

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def forward(self, cgm, insulin, physiology):
        mu_list, lv_list = [], []
        for enc, x in [(self.cgm_enc, cgm), (self.insulin_enc, insulin)]:
            mu, lv = enc(x); mu_list.append(mu); lv_list.append(lv)
        h = self.phys_enc(physiology)
        mu_list.append(self.phys_mu(h)); lv_list.append(self.phys_logvar(h))

        mu, logvar = self.product_of_experts(mu_list, lv_list)
        z = self.reparameterize(mu, logvar)
        return {
            'cgm_recon':  self.cgm_dec(z),
            'ins_recon':  self.insulin_dec(z),
            'phys_recon': self.phys_dec(z),
            'mu': mu, 'logvar': logvar, 'z': z,
        }


# ── 6. Loss ───────────────────────────────────────────────────────────────────
FREE_BITS = 0.5   # min nats per latent dim — prevents posterior collapse

def vae_loss(out, batch, kl_weight,
             w_cgm=1.0, w_ins=1.0, w_phys=0.5):
    recon = (w_cgm  * F.mse_loss(out['cgm_recon'],  batch['cgm']) +
             w_ins  * F.mse_loss(out['ins_recon'],   batch['insulin']) +
             w_phys * F.mse_loss(out['phys_recon'],  batch['physiology']))

    lv = out['logvar'].clamp(-10, 10)
    kl_per_dim = -0.5 * (1 + lv - out['mu'].pow(2) - lv.exp())
    kl = torch.clamp(kl_per_dim, min=FREE_BITS).mean()

    return recon + kl_weight * kl, recon, kl


def kl_weight(epoch, warmup=KL_WARMUP):
    return min(epoch / warmup, 1.0)


# ── 7. Training loop ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, epoch, device):
    model.train()
    tot_loss = tot_recon = tot_kl = 0.0
    w = kl_weight(epoch)
    for batch in tqdm(loader, desc=f'  Ep {epoch+1} train', leave=False):
        cgm  = batch['cgm'].to(device)
        ins  = batch['insulin'].to(device)
        phys = batch['physiology'].to(device)
        optimizer.zero_grad()
        out = model(cgm, ins, phys)
        loss, recon, kl = vae_loss(out, {'cgm': cgm, 'insulin': ins,
                                         'physiology': phys}, w)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss.item(); tot_recon += recon.item(); tot_kl += kl.item()
    n = len(loader)
    return tot_loss/n, tot_recon/n, tot_kl/n


@torch.no_grad()
def val_epoch(model, loader, epoch, device):
    model.eval()
    tot_loss = tot_recon = tot_kl = 0.0
    w = kl_weight(epoch)
    for batch in loader:
        cgm  = batch['cgm'].to(device)
        ins  = batch['insulin'].to(device)
        phys = batch['physiology'].to(device)
        out = model(cgm, ins, phys)
        loss, recon, kl = vae_loss(out, {'cgm': cgm, 'insulin': ins,
                                         'physiology': phys}, w)
        tot_loss += loss.item(); tot_recon += recon.item(); tot_kl += kl.item()
    n = len(loader)
    return tot_loss/n, tot_recon/n, tot_kl/n


def save_plot(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = range(1, len(history['train_loss']) + 1)
    for ax, key, title in zip(axes,
            ['loss', 'recon', 'kl'],
            ['Total loss', 'Reconstruction loss', 'KL divergence']):
        ax.plot(ep, history[f'train_{key}'], label='train')
        ax.plot(ep, history[f'val_{key}'],   label='val', ls='--')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend()
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=100); plt.close(fig)


# ── 8. Main ───────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)

    # --- Preprocessing ---
    print("\n=== Preprocessing ===")
    run_preprocessing()

    # --- Dataset ---
    print("\n=== Loading datasets ===")
    train_ds = MetaboNetDataset(PROC_DIR, 'train')
    test_ds  = MetaboNetDataset(PROC_DIR, 'test')
    val_size  = int(0.15 * len(train_ds))
    train_ds, val_ds = random_split(train_ds, [len(train_ds) - val_size, val_size])
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE=='cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE=='cuda'))

    # --- Model ---
    print("\n=== Model ===")
    model = MultimodalVAE(seq_len=288, physio_dim=PHYSIO_DIM,
                          enc_ch=ENC_CHANNELS, dec_ch=DEC_CHANNELS,
                          latent_dim=LATENT_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  |  Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # --- Training ---
    print("\n=== Training ===")
    history = {k: [] for k in ('train_loss','train_recon','train_kl',
                                'val_loss',  'val_recon',  'val_kl')}
    best_val = float('inf')

    for epoch in range(NUM_EPOCHS):
        tl, tr, tk = train_epoch(model, train_loader, optimizer, epoch, DEVICE)
        vl, vr, vk = val_epoch(model,  val_loader,   epoch,     DEVICE)

        for key, val in zip(('train_loss','train_recon','train_kl',
                              'val_loss',  'val_recon',  'val_kl'),
                             (tl, tr, tk, vl, vr, vk)):
            history[key].append(val)

        w = kl_weight(epoch)
        print(f"Ep {epoch+1:3d}/{NUM_EPOCHS} | "
              f"train {tl:.4f} (recon {tr:.4f} kl {tk:.4f}) | "
              f"val {vl:.4f} (recon {vr:.4f} kl {vk:.4f}) | kl_w {w:.2f}")

        # Save plot every epoch
        save_plot(history, MODEL_DIR / 'loss_curves.png')

        # Checkpoint
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'history': history},
                       MODEL_DIR / f'checkpoint_epoch_{epoch+1:03d}.pt')

        # Best model
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), MODEL_DIR / 'best_model.pt')
            print(f"  -> New best val loss: {best_val:.4f}")

    # --- Save history ---
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Outputs in {MODEL_DIR}")


if __name__ == '__main__':
    main()
