# Getting Started Guide

This guide will help you get up and running with the Diabetes VAE project.

## Quick Installation

### 1. Clone and Install Dependencies

```bash
# Clone (or download) the repository
cd diabetes-vae

# Install required packages
pip install -r requirements.txt

# (Optional) Install development tools
pip install -r requirements-dev.txt
```

### 2. Install Package (for easier imports)

```bash
pip install -e .
```

This allows you to import the package from anywhere: `from src import config, models, ...`

## Running the Quickstart

Get a quick overview of all components in 5 minutes:

```bash
python scripts/quickstart.py
```

This will:
- Demonstrate device setup (CPU/GPU detection)
- Show how to create synthetic data
- Display baseline and multimodal VAE architectures
- Compute loss functions
- Explain next steps

## Training Models

### Option 1: Baseline VAE (CGM only)

```bash
python scripts/train_baseline_vae.py
```

Results saved to: `experiments/baseline_vae/`

### Option 2: Multimodal VAE with Product-of-Experts

```bash
python scripts/train_multimodal_vae.py
```

Results saved to: `experiments/multimodal_vae/`

### Using Custom Configuration

Create a custom config file `my_config.yaml`:

```yaml
model:
  latent_dim: 16
  cgm_encoder_channels: [32, 64, 128]

training:
  batch_size: 64
  learning_rate: 0.0005
  num_epochs: 200
```

Then in your training script:

```python
from src.config import Config
config = Config.from_yaml('my_config.yaml')
```

## Loading and Using Trained Models

```python
import torch
from src.models import MultimodalVAEWithPoE
from src.utils import load_model, get_device

device = get_device()

# Create model
model = MultimodalVAEWithPoE(latent_dim=8)
model.to(device)

# Load trained weights
load_model(model, 'models/best_model.pt', device=device)

# Encode a batch
cgm = torch.randn(4, 288).to(device)
insulin = torch.randn(4, 288).to(device)
physiology = torch.randn(4, 5).to(device)

latent_codes = model.encode(cgm=cgm, insulin=insulin, physiology=physiology)
print(latent_codes.shape)  # (4, 8)

# Decode
reconstruction = model.decode_cgm(latent_codes)
print(reconstruction.shape)  # (4, 288)
```

## Loading Your Own Data

### Step 1: Implement Data Loading

Edit `src/dataset.py` and implement `OhioT1DMDataset` or `MetaboNetDataset`:

```python
class OhioT1DMDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        # Load your patient data here
        # Expected: CGM (288,), Insulin (288,), Physiology (5,)
        pass
    
    def __getitem__(self, idx):
        return {
            'cgm': cgm_sample,           # (288,)
            'insulin': insulin_sample,   # (288,)
            'physiology': phys_sample,   # (5,)
            'patient_id': patient_id,
            'date': date_string,
            'time_in_range': float,
            'glucose_variability': float,
        }
```

### Step 2: Use in Training Scripts

```python
from src.dataset import OhioT1DMDataset
from torch.utils.data import DataLoader

dataset = OhioT1DMDataset(data_dir='/path/to/ohio/data', split='train')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Now train as usual with train_loader
```

## Evaluating Models

### Compute Metrics and Visualize

```bash
python scripts/evaluate_models.py
```

This generates:
- `results/latent_space_umap.png` - UMAP visualization colored by clinical metrics
- Reconstruction errors per modality
- Inter-patient similarity correlation

### Custom Evaluation in Python

```python
from src.evaluation import LatentSpaceEvaluator
from torch.utils.data import DataLoader

evaluator = LatentSpaceEvaluator(output_dir='results')

# Get latent codes for all test samples
latent_codes, metadata = evaluator.get_latent_representations(
    model=trained_model,
    dataloader=test_loader,
    device='cuda'
)

# Visualize with UMAP
evaluator.visualize_latent_space_umap(latent_codes, metadata)

# Compute reconstruction quality
recon_metrics = evaluator.compute_reconstruction_quality(
    model=trained_model,
    dataloader=test_loader,
    device='cuda'
)
print(recon_metrics)

# Analyze clinical similarity (RQ2)
similarity = evaluator.compute_inter_patient_similarity(latent_codes, metadata)
print(f"Correlation between latent distance and clinical similarity: {similarity}")
```

## Customizing the Models

### Changing Architecture

```python
from src.models import BaselineVAE

model = BaselineVAE(
    cgm_length=288,
    encoder_channels=[32, 64, 128, 256],  # Deeper network
    decoder_channels=[256, 128, 64, 32],
    latent_dim=16                          # Higher latent dimension
)
```

### Adjusting Training Hyperparameters

```python
from src.config import Config, TrainingConfig

config = Config()
config.training.learning_rate = 0.0001
config.training.batch_size = 128
config.training.kl_warmup_epochs = 20
config.training.num_epochs = 500
config.model.latent_dim = 32
```

## Common Issues and Solutions

### GPU Out of Memory

Reduce `batch_size`:
```yaml
training:
  batch_size: 16  # Instead of 32
```

### Model Not Learning (Loss Not Decreasing)

1. Check learning rate - try 1e-4 or 1e-3
2. Ensure data normalization is happening (check `normalize=True` in dataset)
3. Verify data shapes match model expectations
4. Check device: is GPU being used? Run `nvidia-smi`

### Posterior Collapse (KL divergence → 0)

Increase `kl_warmup_epochs`:
```yaml
training:
  kl_warmup_epochs: 20
  kl_annealing_enabled: true
```

### Data Dimension Mismatch

Ensure your data loader returns:
- `cgm`: shape `(batch_size, 288)`
- `insulin`: shape `(batch_size, 288)`
- `physiology`: shape `(batch_size, n_features)` or `None`

## Monitoring Training

### Using TensorBoard (Optional)

```bash
pip install tensorboard

# In your training loop (modify training.py):
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)

# View in browser:
tensorboard --logdir=runs/
# Open http://localhost:6006
```

### Tracking History

Training automatically saves `training_history.json` with all metrics:

```python
import json
import matplotlib.pyplot as plt

with open('experiments/baseline_vae/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'], label='val')
plt.ylabel('Total Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['train_recon'], label='train')
plt.plot(history['val_recon'], label='val')
plt.ylabel('Reconstruction Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history['train_kl'], label='train')
plt.plot(history['val_kl'], label='val')
plt.ylabel('KL Divergence')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
```

## Project Workflow

Typical workflow for a complete experiment:

```
1. Data Preparation
   └─> Implement OhioT1DMDataset or MetaboNetDataset

2. Configuration
   └─> Create or modify config.yaml

3. Training
   ├─> python scripts/train_baseline_vae.py
   └─> python scripts/train_multimodal_vae.py

4. Evaluation
   └─> python scripts/evaluate_models.py

5. Analysis
   ├─> UMAP visualization
   ├─> Latent trajectories (RQ1)
   └─> Inter-patient similarity (RQ2)

6. Hyperparameter Tuning (optional)
   └─> Repeat 2-5 with different configs
```

## Research Questions

### RQ1: Drift Detection

Analyze patient trajectories in latent space:

```python
# In evaluation.py or custom script
# For each patient, plot z(t) over time
# Correlate displacement with clinical changes
```

### RQ2: Inter-Patient Similarity

Check if similar patients cluster:

```python
# Already computed in evaluate_models.py
# Look for positive correlation between:
# - Latent space distance
# - Clinical differences (TIR, glucose variability)
```

## Resources

- **VAE Tutorial**: https://arxiv.org/abs/1312.6114
- **Product-of-Experts**: https://arxiv.org/abs/1802.07998
- **Multimodal Learning**: https://arxiv.org/abs/2209.03430
- **PyTorch Docs**: https://pytorch.org/docs/stable/

## Next Steps

1. ✅ Run `python scripts/quickstart.py`
2. 🔄 Prepare your data
3. 🚀 Train baseline model
4. 📊 Evaluate and visualize
5. 🔧 Tune hyperparameters
6. 📈 Compare with baselines
7. 📝 Write results!

## Need Help?

- Check the code docstrings: Each class and function has detailed documentation
- See example usage in the training scripts
- Review `src/config.py` for all available settings
- Check error messages carefully - they're usually informative!

Good luck! 🚀
