# Diabetes VAE: Latent Metabolic State Learning with Variational Autoencoders

A research project implementing variational autoencoders (VAEs) for learning latent metabolic states from continuous glucose monitoring (CGM) and wearable data.

## Project Overview

This project addresses two key research questions:

**RQ1 - Drift:** Does a patient's position in the learned latent space change meaningfully over time, and does that drift correlate with clinically observable changes in metabolic control?

**RQ2 - Similarity:** Does distance between patients in the latent space reflect clinical similarity, measured by proxies such as time-in-range, glucose variability, and HbA1c?

## Dataset

The project uses two complementary datasets:

- **OhioT1DM** (primary development): 12 patients with type 1 diabetes over 8 weeks, with rich multimodal data (CGM, insulin, meals, exercise, wearables). Requires Data Use Agreement from Ohio University.

- **MetaboNet** (scale-up/validation): >3,100 patients and 1,228 patient-years of CGM + insulin data. Public subset available at metabo-net.org.

## Model Architecture

### Baseline VAE
Standard 1D convolutional VAE on CGM data only, establishing a baseline for glucose pattern learning.

### Multimodal VAE with Product-of-Experts (PoE)
Advanced architecture combining:
- Separate 1D CNN encoders for CGM and insulin profiles
- MLP encoder for daily physiology summaries
- Product-of-Experts fusion for principled missing data handling
- Shared decoders for all modalities

### Optional: Hierarchical VAE
Temporal latent variable aggregating daily states across weeks, enabling analysis of metabolic drift.

## Project Structure

```
diabetes-vae/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── dataset.py               # Dataset classes and utilities
│   ├── models.py                # VAE model implementations
│   ├── training.py              # Training pipeline with KL annealing
│   ├── evaluation.py            # Evaluation metrics and visualization
│   └── utils.py                 # Utility functions
├── scripts/
│   ├── train_baseline_vae.py    # Training script for baseline model
│   ├── train_multimodal_vae.py  # Training script for multimodal model
│   └── evaluate_models.py       # Evaluation and analysis script
├── data/                        # Data directory (placeholder)
├── experiments/                 # Experiment outputs and logs
├── results/                     # Results and visualizations
├── models/                      # Trained models
├── config.yaml                  # Default configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) For GPU acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Training Baseline VAE

```bash
python scripts/train_baseline_vae.py
```

This trains a standard VAE on CGM data with KL annealing and saves checkpoints to `experiments/baseline_vae/models/`.

### Training Multimodal VAE

```bash
python scripts/train_multimodal_vae.py
```

This trains the multimodal VAE with Product-of-Experts fusion on CGM, insulin, and physiology data.

### Evaluation

```bash
python scripts/evaluate_models.py
```

Evaluates trained models and generates:
- UMAP visualizations of latent space
- Reconstruction quality metrics
- Inter-patient similarity analysis

## Configuration

Edit `config.yaml` to customize:
- Model architecture (latent dimension, encoder/decoder channels)
- Training hyperparameters (learning rate, batch size, KL annealing schedule)
- Data handling and preprocessing
- Evaluation metrics

Or create a new YAML config and pass it to training scripts.

## Evaluation Metrics

### Latent Space Structure
- **UMAP visualization**: Color by clinical metrics (time-in-range, glucose variability, patient ID)
- **Clustering quality**: Similar days should cluster together without explicit labels

### Drift Detection (RQ1)
- Plot per-patient latent trajectories z(t) over time
- Correlate latent space displacement with changes in weekly clinical metrics

### Inter-Patient Similarity (RQ2)
- Compute pairwise distances in latent space
- Compare with pairwise clinical differences
- Report Spearman rank correlation

### Missing Modality Robustness
- Systematically mask modalities at test time
- Measure reconstruction quality and latent stability
- Compare graceful degradation vs concatenation baselines

## Key Features

- **Principled missing data handling**: Product-of-Experts fusion degrades gracefully when modalities are missing
- **KL annealing**: Prevents posterior collapse during early training
- **Multiple random seeds**: All metrics reported as mean ± std
- **Modular design**: Easy to extend with new modalities or architectural variants

## Training Strategy

1. **KL Annealing**: Linearly increase KL weight from 0 to 1 over first 10 epochs to prevent posterior collapse
2. **Separate monitoring**: Track reconstruction and KL terms independently
3. **Checkpointing**: Save model every 5 epochs
4. **Validation-based model selection**: Keep best model based on validation loss

## Hyperparameter Tuning

The project is designed to enable full hyperparameter sweeps on high-end GPUs (RTX 5070+):

Key hyperparameters to tune:
- `latent_dim`: 4, 8, 16, 32
- `learning_rate`: 1e-4, 1e-3, 1e-2
- `batch_size`: 16, 32, 64
- `kl_warmup_epochs`: 5, 10, 20
- `modality_weights`: Equal, or weighted by importance

## Data Format

Expected data format for each patient-day sample:
- **CGM**: 288 glucose readings (one every 5 minutes, 24 hours)
- **Insulin**: 288 insulin doses (bolus + basal, same 5-min grid)
- **Physiology**: Daily summaries of heart rate, steps, skin signals (e.g., GSR, skin temperature)

See `src/dataset.py` for placeholder data loading functions.

## References

- Kingma & Welling (2013): Auto-Encoding Variational Bayes
- Shi et al. (2019): Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models
- Product-of-Experts: Hinton (2002)

## Citation

If you use this project, please cite:

```
@inproceedings{diabetes_vae_2024,
  title={Latent Metabolic State Learning with Variational Autoencoders},
  author={[Your Name]},
  booktitle={[Venue]},
  year={2024}
}
```

## License

See [LICENSE](LICENSE) for details.

## Contact & Support

For questions or issues:
- Open an issue on GitHub
- Contact: [your email]

## Future Work

- [ ] Hierarchical VAE for temporal aggregation
- [ ] Attention mechanisms for modality importance
- [ ] Temporal VAE for sequence modeling
- [ ] Semi-supervised learning with clinical labels
- [ ] Anomaly detection on latent trajectories
- [ ] Interactive visualization dashboard