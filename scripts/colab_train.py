"""
Google Colab GPU training script — Diabetes Multimodal VAE
===========================================================

SETUP (run once before executing this script):
-----------------------------------------------
  Cell 1 — Mount Drive and install deps:
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install -q pyarrow umap-learn

  Cell 2 — Upload parquet files to Drive (do this from your computer):
    Place in:  My Drive/metabonet/train.parquet
               My Drive/metabonet/test.parquet

  Cell 3 — Enable GPU:
    Runtime > Change runtime type > T4 GPU

  Cell 4 — Run this script:
    exec(open('/content/drive/MyDrive/diabetes-vae/scripts/colab_train.py').read())
    main()

What gets saved to Drive (persistent across sessions):
  My Drive/diabetes-vae/processed/{train,test}/   ← preprocessed arrays
  My Drive/diabetes-vae/models/best_model.pt
  My Drive/diabetes-vae/models/checkpoint_epoch_*.pt
  My Drive/diabetes-vae/models/training_history.json
  My Drive/diabetes-vae/models/loss_curves.png

Resuming after a disconnect:
  Just re-run the script — it auto-resumes from the latest checkpoint
  and skips preprocessing if already done.
"""

# ── 0. Imports & environment detection ───────────────────────────────────────
import os, sys, json, shutil, subprocess
from pathlib import Path

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

try:                                      # show plots inline on Colab
    from IPython.display import display, Image as IPImage
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

ON_COLAB  = 'google.colab' in sys.modules
ON_KAGGLE = Path('/kaggle').exists() and not ON_COLAB

# ── 1. Paths ──────────────────────────────────────────────────────────────────
if ON_COLAB:
    DRIVE_DIR = Path('/content/drive/MyDrive/diabetes-vae')
    RAW_DIR   = Path('/content/drive/MyDrive/metabonet')   # parquet files here
    # Processed arrays live on Drive (persistent), but are copied to /content/
    # for fast DataLoader access during training
    PROC_DRIVE = DRIVE_DIR / 'processed'
    PROC_LOCAL = Path('/content/processed')                  # fast local copy
    MODEL_DIR  = DRIVE_DIR / 'models'
elif ON_KAGGLE:
    RAW_DIR    = Path('/kaggle/input/metabonet-diabetes')
    PROC_DRIVE = None
    PROC_LOCAL = Path('/kaggle/working/processed')
    MODEL_DIR  = Path('/kaggle/working/models')
else:
    RAW_DIR    = Path('data')
    PROC_DRIVE = None
    PROC_LOCAL = Path('data/processed')
    MODEL_DIR  = Path('models')

PROC_DIR = PROC_LOCAL   # used everywhere after setup
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── 2. Hyperparameters ────────────────────────────────────────────────────────
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS     = 2 if (ON_COLAB or ON_KAGGLE) else 0
BATCH_SIZE      = 128
NUM_EPOCHS      = 100
LR              = 3e-4
KL_WARMUP       = 25
LATENT_DIM      = 8
ENC_CHANNELS    = [16, 32, 64]
DEC_CHANNELS    = [64, 32, 16]
PHYSIO_DIM      = 5
MIN_CGM_FRAC    = 0.5
CHECKPOINT_FREQ = 5
FREE_BITS       = 0.5    # min nats/dim — prevents posterior collapse
SEED            = 42

print(f"Environment : {'Colab' if ON_COLAB else 'Kaggle' if ON_KAGGLE else 'local'}")
print(f"Device      : {DEVICE}" +
      (f"  ({torch.cuda.get_device_name(0)})" if DEVICE == 'cuda' else ''))
print(f"Outputs     : {MODEL_DIR}")


# ── 3. Preprocessing ──────────────────────────────────────────────────────────
PHYSIO_COLS = ['heartrate', 'steps', 'galvanic_skin_response',
               'skin_temp', 'calories_burned']
SLOTS = 288

def slot_index(ts):
    return (ts.dt.hour * 60 + ts.dt.minute) // 5

def interpolate_cgm(arr):
    s = pd.Series(arr).interpolate(method='linear', limit=12).ffill().bfill()
    return s.values.astype(np.float32)

def preprocess_split(parquet_path, out_dir, batch_size=1_000_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    days = {}
    pf   = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    print(f"  {parquet_path.name}: {total:,} rows")

    for batch in tqdm(pf.iter_batches(batch_size=batch_size),
                      total=total // batch_size + 1, desc='  reading'):
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
                    'n_cgm': 0, 'patient_id': pid, 'date': str(day),
                    'source': r.get('source_file', ''),
                    'gender': r.get('gender', None),
                    'age':    r.get('age', np.nan),
                }
            e     = days[key]
            slots = grp['_slot'].values.astype(int)

            cgm_v = grp['CGM'].values.astype(np.float32)
            ok = ~np.isnan(cgm_v)
            if ok.any():
                e['cgm'][slots[ok]] = cgm_v[ok]
                e['n_cgm'] += int(ok.sum())

            ins_v = grp['insulin'].values
            ok2 = ~np.isnan(ins_v)
            if ok2.any():
                e['insulin'][slots[ok2]] = ins_v[ok2].astype(np.float32)

            for j, col in enumerate(PHYSIO_COLS):
                if col in grp.columns:
                    vals = grp[col].dropna().values
                    if len(vals):
                        e['phys_sum'][j]   += vals.sum()
                        e['phys_count'][j] += len(vals)

    min_cgm = int(MIN_CGM_FRAC * SLOTS)
    kept    = [v for v in days.values() if v['n_cgm'] >= min_cgm]
    print(f"  Patient-days: {len(days):,} → {len(kept):,} after filter")

    cgm_arr  = np.stack([interpolate_cgm(e['cgm']) for e in kept])
    ins_arr  = np.stack([e['insulin']               for e in kept])

    phys_arr = np.zeros((len(kept), len(PHYSIO_COLS)), dtype=np.float32)
    for i, e in enumerate(kept):
        mask = e['phys_count'] > 0
        with np.errstate(invalid='ignore'):
            phys_arr[i, mask] = (e['phys_sum'][mask] /
                                 e['phys_count'][mask]).astype(np.float32)
        phys_arr[i, ~mask] = np.nan

    meta = pd.DataFrame({k: [e[k] for e in kept]
                         for k in ('patient_id','date','source',
                                   'gender','age','n_cgm')})
    np.save(out_dir / 'cgm.npy',        cgm_arr)
    np.save(out_dir / 'insulin.npy',    ins_arr)
    np.save(out_dir / 'physiology.npy', phys_arr)
    meta.to_parquet(out_dir / 'metadata.parquet', index=False)
    print(f"  Saved {cgm_arr.shape} to {out_dir}")


def setup_processed_data():
    """Preprocess if needed; copy from Drive to local for fast I/O."""
    for split in ('train', 'test'):
        local_out = PROC_LOCAL / split
        drive_out = (PROC_DRIVE / split) if PROC_DRIVE else None

        # Already local → done
        if (local_out / 'cgm.npy').exists():
            print(f"'{split}' already local, skipping.")
            continue

        # On Drive but not local → copy
        if drive_out and (drive_out / 'cgm.npy').exists():
            print(f"Copying '{split}' from Drive to /content/ for fast I/O...")
            shutil.copytree(drive_out, local_out)
            continue

        # Need to preprocess from raw parquet
        src = RAW_DIR / f'{split}.parquet'
        if not src.exists():
            raise FileNotFoundError(
                f"Not found: {src}\n"
                "Upload train.parquet / test.parquet to My Drive/metabonet/")
        print(f"\nPreprocessing {split}...")
        preprocess_split(src, local_out)

        # Mirror to Drive for next session
        if drive_out:
            print(f"Copying to Drive for persistence...")
            shutil.copytree(local_out, drive_out)


# ── 4. Dataset ────────────────────────────────────────────────────────────────
class MetaboNetDataset(Dataset):
    def __init__(self, split, normalize=True):
        d = PROC_DIR / split
        self.cgm  = np.load(d / 'cgm.npy',        mmap_mode='r')
        self.ins  = np.load(d / 'insulin.npy',     mmap_mode='r')
        self.phys = np.load(d / 'physiology.npy',  mmap_mode='r')
        self.meta = pd.read_parquet(d / 'metadata.parquet')
        self.n    = len(self.cgm)
        self.sc   = {}
        if normalize:
            self._fit_scalers()

    def _fit_scalers(self):
        cgm  = np.array(self.cgm,  dtype=np.float32)
        ins  = np.array(self.ins,  dtype=np.float32)
        phys = np.array(self.phys, dtype=np.float32)
        phys_obs = phys[~np.isnan(phys).any(axis=1)]
        self.sc['cgm']     = StandardScaler().fit(cgm)
        self.sc['insulin'] = StandardScaler().fit(ins)
        if len(phys_obs):
            self.sc['physiology'] = StandardScaler().fit(phys_obs)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        cgm  = np.array(self.cgm[idx],  dtype=np.float32)
        ins  = np.array(self.ins[idx],   dtype=np.float32)
        phys = np.array(self.phys[idx],  dtype=np.float32)
        if 'cgm'     in self.sc: cgm  = self.sc['cgm'].transform(cgm.reshape(1,-1)).flatten()
        if 'insulin' in self.sc: ins  = self.sc['insulin'].transform(ins.reshape(1,-1)).flatten()
        if np.isnan(phys).any():
            phys = np.zeros_like(phys)
        elif 'physiology' in self.sc:
            phys = self.sc['physiology'].transform(phys.reshape(1,-1)).flatten()
        row = self.meta.iloc[idx]
        return {'cgm': torch.from_numpy(cgm), 'insulin': torch.from_numpy(ins),
                'physiology': torch.from_numpy(phys),
                'patient_id': row['patient_id'], 'date': row['date']}


# ── 5. Model ──────────────────────────────────────────────────────────────────
class ConvEncoder1D(nn.Module):
    def __init__(self, input_length, channels, latent_dim=8, kernel_size=3, padding=1):
        super().__init__()
        layers, in_ch = [], 1
        for out_ch in channels:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding), nn.ReLU()]
            in_ch = out_ch
        self.conv      = nn.Sequential(*layers)
        self.fc_mu     = nn.Linear(in_ch * input_length, latent_dim)
        self.fc_logvar = nn.Linear(in_ch * input_length, latent_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder1D(nn.Module):
    def __init__(self, output_length, channels, latent_dim=8, kernel_size=3, padding=1):
        super().__init__()
        self.L  = output_length
        self.C0 = channels[0]
        self.fc = nn.Linear(latent_dim, channels[0] * output_length)
        layers, in_ch = [], channels[0]
        for out_ch in channels[1:] + [1]:
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size, padding=padding))
            if out_ch != 1: layers.append(nn.ReLU())
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        return self.deconv(self.fc(z).view(z.size(0), self.C0, self.L)).squeeze(1)


class MultimodalVAE(nn.Module):
    def __init__(self, seq_len=288, physio_dim=5, enc_ch=None,
                 dec_ch=None, latent_dim=8):
        super().__init__()
        enc_ch = enc_ch or ENC_CHANNELS
        dec_ch = dec_ch or DEC_CHANNELS
        self.latent_dim = latent_dim
        self.cgm_enc     = ConvEncoder1D(seq_len, enc_ch, latent_dim)
        self.insulin_enc = ConvEncoder1D(seq_len, enc_ch, latent_dim)
        self.phys_enc    = nn.Sequential(nn.Linear(physio_dim,64),nn.ReLU(),
                                         nn.Linear(64,32),nn.ReLU())
        self.phys_mu     = nn.Linear(32, latent_dim)
        self.phys_logvar = nn.Linear(32, latent_dim)
        self.cgm_dec     = ConvDecoder1D(seq_len, dec_ch, latent_dim)
        self.insulin_dec = ConvDecoder1D(seq_len, dec_ch, latent_dim)
        self.phys_dec    = nn.Sequential(nn.Linear(latent_dim,32),nn.ReLU(),
                                         nn.Linear(32,64),nn.ReLU(),
                                         nn.Linear(64,physio_dim))

    def product_of_experts(self, mu_list, lv_list):
        prec = torch.zeros_like(mu_list[0])
        wmu  = torch.zeros_like(mu_list[0])
        for mu, lv in zip(mu_list, lv_list):
            p = 1.0 / (lv.clamp(-10,10).exp() + 1e-8)
            prec += p; wmu += p * mu
        mu_poe  = wmu / (prec + 1e-8)
        lv_poe  = torch.log(1.0 / (prec + 1e-8) + 1e-8)
        return mu_poe, lv_poe

    def reparameterize(self, mu, lv):
        return mu + torch.randn_like(mu) * (0.5 * lv).exp()

    def forward(self, cgm, ins, phys):
        mu_l, lv_l = [], []
        for enc, x in [(self.cgm_enc, cgm), (self.insulin_enc, ins)]:
            m, l = enc(x); mu_l.append(m); lv_l.append(l)
        h = self.phys_enc(phys)
        mu_l.append(self.phys_mu(h)); lv_l.append(self.phys_logvar(h))
        mu, lv = self.product_of_experts(mu_l, lv_l)
        z = self.reparameterize(mu, lv)
        return {'cgm_recon':  self.cgm_dec(z),
                'ins_recon':  self.insulin_dec(z),
                'phys_recon': self.phys_dec(z),
                'mu': mu, 'logvar': lv, 'z': z}


# ── 6. Loss & schedule ────────────────────────────────────────────────────────
def kl_weight(epoch): return min(epoch / KL_WARMUP, 1.0)

def vae_loss(out, cgm, ins, phys, epoch):
    recon = (F.mse_loss(out['cgm_recon'],  cgm)  +
             F.mse_loss(out['ins_recon'],   ins)  +
         0.5*F.mse_loss(out['phys_recon'],  phys))
    lv  = out['logvar'].clamp(-10, 10)
    kl  = torch.clamp(-0.5*(1 + lv - out['mu'].pow(2) - lv.exp()),
                      min=FREE_BITS).mean()
    w   = kl_weight(epoch)
    return recon + w * kl, recon, kl, w


# ── 7. Plotting (inline + saved to Drive) ────────────────────────────────────
def plot_and_save(history, epoch):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Training progress — epoch {epoch+1}/{NUM_EPOCHS}',
                 fontsize=12, fontweight='bold')
    ep = range(1, len(history['train_loss']) + 1)
    for ax, key, title in zip(axes,
            ['loss','recon','kl'],
            ['Total loss','Reconstruction loss','KL divergence']):
        ax.plot(ep, history[f'train_{key}'], lw=2, label='train')
        ax.plot(ep, history[f'val_{key}'],   lw=2, label='val', ls='--')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend()
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    plot_path = MODEL_DIR / 'loss_curves.png'
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    if IN_NOTEBOOK:
        display(fig)    # show inline in Colab cell
    plt.close(fig)


# ── 8. Checkpoint helpers ─────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, history):
    path = MODEL_DIR / f'checkpoint_epoch_{epoch+1:03d}.pt'
    torch.save({'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history}, path)
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    return path


def find_latest_checkpoint():
    ckpts = sorted(MODEL_DIR.glob('checkpoint_epoch_*.pt'))
    return ckpts[-1] if ckpts else None


def load_checkpoint(model, optimizer):
    ckpt_path = find_latest_checkpoint()
    if ckpt_path is None:
        print("No checkpoint found — starting from scratch.")
        return 0, {k: [] for k in ('train_loss','train_recon','train_kl',
                                    'val_loss',  'val_recon',  'val_kl')}
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch'] + 1
    history     = ckpt['history']
    print(f"Resumed from {ckpt_path.name}  (epoch {start_epoch}/{NUM_EPOCHS})")
    return start_epoch, history


# ── 9. Training loop ──────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, epoch, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_recon = tot_kl = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=f"  {'train' if train else 'val ':4s}", leave=False):
            cgm  = batch['cgm'].to(DEVICE)
            ins  = batch['insulin'].to(DEVICE)
            phys = batch['physiology'].to(DEVICE)
            if train: optimizer.zero_grad()
            out = model(cgm, ins, phys)
            loss, recon, kl, w = vae_loss(out, cgm, ins, phys, epoch)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot_loss += loss.item(); tot_recon += recon.item(); tot_kl += kl.item()
    n = len(loader)
    return tot_loss/n, tot_recon/n, tot_kl/n


# ── 10. Main ──────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)

    # --- Data ---
    print("\n=== Data setup ===")
    setup_processed_data()

    print("\n=== Loading datasets ===")
    train_ds = MetaboNetDataset('train')
    test_ds  = MetaboNetDataset('test')
    val_n    = int(0.15 * len(train_ds))
    train_ds, val_ds = random_split(train_ds, [len(train_ds) - val_n, val_n])
    print(f"Train {len(train_ds):,}  |  Val {len(val_ds):,}  |  Test {len(test_ds):,}")

    pin = DEVICE == 'cuda'
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # --- Model ---
    print("\n=== Model ===")
    model = MultimodalVAE(seq_len=288, physio_dim=PHYSIO_DIM,
                          latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Resume or start fresh ---
    start_epoch, history = load_checkpoint(model, optimizer)
    if start_epoch >= NUM_EPOCHS:
        print("Training already complete.")
        return

    best_val = min(history['val_loss']) if history['val_loss'] else float('inf')

    # --- Training loop ---
    print(f"\n=== Training (epochs {start_epoch+1}–{NUM_EPOCHS}) ===")
    for epoch in range(start_epoch, NUM_EPOCHS):
        tl, tr, tk = run_epoch(model, train_loader, optimizer, epoch, train=True)
        vl, vr, vk = run_epoch(model, val_loader,   optimizer, epoch, train=False)

        for k, v in zip(('train_loss','train_recon','train_kl',
                          'val_loss',  'val_recon',  'val_kl'),
                         (tl, tr, tk, vl, vr, vk)):
            history[k].append(v)

        w = kl_weight(epoch)
        print(f"Ep {epoch+1:3d}/{NUM_EPOCHS} | "
              f"train {tl:.4f} (r {tr:.4f} kl {tk:.4f}) | "
              f"val {vl:.4f} (r {vr:.4f} kl {vk:.4f}) | kl_w {w:.2f}")

        # Plot (inline + saved to Drive every epoch)
        plot_and_save(history, epoch)

        # Checkpoint every N epochs
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            ckpt = save_checkpoint(model, optimizer, epoch, history)
            print(f"  Checkpoint saved: {ckpt.name}")

        # Best model
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), MODEL_DIR / 'best_model.pt')
            print(f"  New best: {best_val:.4f}")

    # Final save
    save_checkpoint(model, optimizer, NUM_EPOCHS - 1, history)
    print(f"\nDone. All outputs in {MODEL_DIR}")
