"""
Project summary figure: dataset pipeline, model architecture, and training design.
Run from the project root:
    python scripts/make_summary.py
Output: results/project_summary.png
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

COLORS = {
    'cgm':        '#2196F3',
    'insulin':    '#FF9800',
    'physiology': '#4CAF50',
    'latent':     '#9C27B0',
    'decoder':    '#F44336',
    'data':       '#E3F2FD',
    'model':      '#F3E5F5',
    'train':      '#1976D2',
    'val':        '#E53935',
    'neutral':    '#F5F5F5',
    'border':     '#BDBDBD',
}


# ── helpers ──────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel=None, color='#E3F2FD', fontsize=9,
        border='#90CAF9', bold=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle='round,pad=0.02', linewidth=1.2,
                          edgecolor=border, facecolor=color, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    yo = y + 0.04 if sublabel else y
    ax.text(x, yo, label, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, zorder=4)
    if sublabel:
        ax.text(x, y - 0.12, sublabel, ha='center', va='center',
                fontsize=fontsize - 1.5, color='#555', zorder=4)


def arrow(ax, x0, y0, x1, y1, color='#555', lw=1.2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'))


# ── Panel A: Dataset pipeline ─────────────────────────────────────────────

def panel_pipeline(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('A  |  Dataset Pipeline', fontsize=11, fontweight='bold',
                 loc='left', pad=8)

    # Raw files
    box(ax, 2, 5.0, 3.2, 0.65, 'train.parquet',  '132 M rows · 36 columns',
        color='#FFF8E1', border='#FFB300')
    box(ax, 2, 4.1, 3.2, 0.65, 'test.parquet',   '22 M rows · 36 columns',
        color='#FFF8E1', border='#FFB300')

    arrow(ax, 3.6, 5.0,  5.2, 3.55)
    arrow(ax, 3.6, 4.1,  5.2, 3.45)

    # Preprocessing
    box(ax, 6.2, 3.5, 2.8, 0.85, 'preprocess_metabonet.py',
        'stream · group · interpolate · filter',
        color='#E8F5E9', border='#43A047', fontsize=8.5)

    arrow(ax, 7.6, 3.5, 8.5, 3.5)

    # Processed arrays
    box(ax, 9.1, 4.1, 1.6, 0.55, 'cgm.npy',       '(N,288)',
        color=COLORS['data'], border=COLORS['cgm'], fontsize=8)
    box(ax, 9.1, 3.5, 1.6, 0.55, 'insulin.npy',   '(N,288)',
        color=COLORS['data'], border=COLORS['insulin'], fontsize=8)
    box(ax, 9.1, 2.9, 1.6, 0.55, 'physiology.npy','(N,5)',
        color=COLORS['data'], border=COLORS['physiology'], fontsize=8)
    box(ax, 9.1, 2.3, 1.6, 0.55, 'metadata',      'patient·date',
        color=COLORS['data'], border=COLORS['border'], fontsize=8)

    arrow(ax, 7.6, 3.5, 8.3, 4.1)
    arrow(ax, 7.6, 3.5, 8.3, 3.5)
    arrow(ax, 7.6, 3.5, 8.3, 2.9)
    arrow(ax, 7.6, 3.5, 8.3, 2.3)

    # Steps annotation
    steps = [
        (0.35, 4.95, '① 5-min slot index'),
        (0.35, 4.45, '② group (patient, day)'),
        (0.35, 3.95, '③ fill CGM array'),
        (0.35, 3.45, '④ accumulate insulin'),
        (0.35, 2.95, '⑤ daily physiology stats'),
        (0.35, 2.45, '⑥ linear interpolation'),
        (0.35, 1.95, '⑦ filter ≥ 50 % CGM'),
    ]
    for x, y, txt in steps:
        ax.text(x, y, txt, fontsize=7.5, va='center', color='#333')

    # Stats box
    stats = (
        'Dataset stats\n'
        '━━━━━━━━━━━━━━━━\n'
        '379,849 patient-days\n'
        '1,167 unique patients\n'
        '13 source studies\n'
        'NaN after processing: 0 %\n'
        'Coverage filter: ≥ 50 %'
    )
    ax.text(5.5, 1.6, stats, fontsize=8, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='#F3E5F5',
                      edgecolor='#9C27B0', linewidth=1.2),
            family='monospace')


# ── Panel B: Source distribution ─────────────────────────────────────────

def panel_sources(ax):
    sources = {
        'Loop':     243039,
        'ReplaceBG': 41238,
        'IOBP2':    24858,
        'DCLP3':    17415,
        'DCLP5':    16507,
        'PEDAP':    15145,
        'Flair':    12431,
        'CTR3':      4183,
        'BrisT1D':   2330,
        'AZT1D':      839,
        'HUPA-UCM':   637,
        'T1D-UOM':    615,
        'OhioT1DM':   612,
    }
    names  = list(sources.keys())
    counts = list(sources.values())
    total  = sum(counts)
    pct    = [c / total * 100 for c in counts]

    palette = plt.cm.tab20(np.linspace(0, 1, len(names)))
    bars = ax.barh(names[::-1], counts[::-1], color=palette[::-1],
                   edgecolor='white', linewidth=0.6)

    for bar, p in zip(bars, pct[::-1]):
        ax.text(bar.get_width() + 1500, bar.get_y() + bar.get_height() / 2,
                f'{p:.1f}%', va='center', fontsize=7.5, color='#444')

    ax.set_xlabel('Patient-days', fontsize=9)
    ax.set_title('B  |  Source Study Distribution', fontsize=11,
                 fontweight='bold', loc='left', pad=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, max(counts) * 1.18)


# ── Panel C: Model architecture ──────────────────────────────────────────

def panel_model(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('C  |  Multimodal VAE with Product-of-Experts',
                 fontsize=11, fontweight='bold', loc='left', pad=8)

    # Inputs
    box(ax, 1.0, 5.8, 1.6, 0.55, 'CGM',       '(batch, 288)',
        color='#E3F2FD', border=COLORS['cgm'])
    box(ax, 1.0, 4.0, 1.6, 0.55, 'Insulin',   '(batch, 288)',
        color='#FFF3E0', border=COLORS['insulin'])
    box(ax, 1.0, 2.2, 1.6, 0.55, 'Physiology','(batch, 5)',
        color='#E8F5E9', border=COLORS['physiology'])

    # Encoders
    box(ax, 3.3, 5.8, 2.0, 0.65, 'Conv1D Encoder',
        'Conv×3 → FC → (μ,σ²)', color='#E3F2FD', border=COLORS['cgm'], fontsize=8)
    box(ax, 3.3, 4.0, 2.0, 0.65, 'Conv1D Encoder',
        'Conv×3 → FC → (μ,σ²)', color='#FFF3E0', border=COLORS['insulin'], fontsize=8)
    box(ax, 3.3, 2.2, 2.0, 0.65, 'MLP Encoder',
        'Linear×2 → (μ,σ²)',    color='#E8F5E9', border=COLORS['physiology'], fontsize=8)

    for y in [5.8, 4.0, 2.2]:
        arrow(ax, 1.8, y, 2.3, y)
        arrow(ax, 4.3, y, 5.1, y if y != 4.0 else 4.0)

    # PoE
    box(ax, 5.8, 4.0, 1.4, 2.5, 'Product\nof\nExperts',
        None, color='#F3E5F5', border=COLORS['latent'], fontsize=9, bold=True)

    # Latent
    box(ax, 7.5, 4.0, 1.4, 0.55, 'z ~ N(μ,σ²)',
        'latent dim = 8', color='#EDE7F6', border=COLORS['latent'], fontsize=8)
    arrow(ax, 6.5, 4.0, 6.8, 4.0)
    arrow(ax, 8.2, 4.0, 8.6, 5.8)
    arrow(ax, 8.2, 4.0, 8.6, 4.0)
    arrow(ax, 8.2, 4.0, 8.6, 2.2)

    # Decoders
    box(ax, 9.2, 5.8, 1.4, 0.55, 'CGM decoder',      'ConvTranspose×3',
        color='#FFEBEE', border=COLORS['decoder'], fontsize=7.5)
    box(ax, 9.2, 4.0, 1.4, 0.55, 'Insulin decoder',  'ConvTranspose×3',
        color='#FFEBEE', border=COLORS['decoder'], fontsize=7.5)
    box(ax, 9.2, 2.2, 1.4, 0.55, 'Physiology dec.',  'Linear×3',
        color='#FFEBEE', border=COLORS['decoder'], fontsize=7.5)

    # Param count
    ax.text(5.0, 0.8,
            '668,279 total parameters  ·  all trainable',
            ha='center', fontsize=8.5, style='italic', color='#555',
            bbox=dict(boxstyle='round', facecolor='#FAFAFA',
                      edgecolor=COLORS['border'], linewidth=1))


# ── Panel D: KL annealing schedule ───────────────────────────────────────

def panel_annealing(ax):
    epochs = np.arange(0, 101)
    warmup = 10
    kl_w   = np.minimum(epochs / warmup, 1.0)

    ax.fill_between(epochs, kl_w, alpha=0.15, color=COLORS['latent'])
    ax.plot(epochs, kl_w, color=COLORS['latent'], lw=2)
    ax.axvline(warmup, ls='--', color='#999', lw=1)
    ax.text(warmup + 1, 0.5, 'warmup ends\n(epoch 10)',
            fontsize=8, color='#666', va='center')

    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('KL weight', fontsize=9)
    ax.set_title('D  |  KL Annealing Schedule', fontsize=11,
                 fontweight='bold', loc='left', pad=8)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, 100)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)

    # Loss formula annotation
    ax.text(55, 0.3,
            r'$\mathcal{L} = \mathcal{L}_{recon} + w_t \cdot \mathcal{L}_{KL}$',
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#F3E5F5',
                      edgecolor=COLORS['latent'], linewidth=1))


# ── Panel E: Smoke-test training history ─────────────────────────────────

def panel_history(ax_loss, ax_kl):
    history_path = Path('experiments/multimodal_vae/training_history.json')
    if not history_path.exists():
        for ax in [ax_loss, ax_kl]:
            ax.text(0.5, 0.5, 'No training history yet',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='#999')
        return

    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h['train_loss']) + 1)

    ax_loss.plot(epochs, h['train_loss'],  color=COLORS['train'], lw=2, label='train total')
    ax_loss.plot(epochs, h['val_loss'],    color=COLORS['val'],   lw=2, label='val total', ls='--')
    ax_loss.plot(epochs, h['train_recon'], color=COLORS['train'], lw=1.2, label='train recon', ls=':')
    ax_loss.plot(epochs, h['val_recon'],   color=COLORS['val'],   lw=1.2, label='val recon',   ls=':')
    ax_loss.set_title('E  |  Smoke-test Loss (3 epochs)', fontsize=11,
                      fontweight='bold', loc='left', pad=8)
    ax_loss.set_xlabel('Epoch', fontsize=9)
    ax_loss.set_ylabel('MSE loss', fontsize=9)
    ax_loss.legend(fontsize=7.5, ncol=2)
    ax_loss.spines[['top', 'right']].set_visible(False)
    ax_loss.tick_params(labelsize=8)

    ax_kl.plot(epochs, h['train_kl'], color=COLORS['train'], lw=2, label='train KL')
    ax_kl.plot(epochs, h['val_kl'],   color=COLORS['val'],   lw=2, label='val KL', ls='--')
    ax_kl.set_title('F  |  KL Divergence (smoke test)', fontsize=11,
                    fontweight='bold', loc='left', pad=8)
    ax_kl.set_xlabel('Epoch', fontsize=9)
    ax_kl.set_ylabel('KL', fontsize=9)
    ax_kl.legend(fontsize=8)
    ax_kl.spines[['top', 'right']].set_visible(False)
    ax_kl.tick_params(labelsize=8)

    note = ('KL annealing weight:\n'
            'ep1=0.0, ep2=0.1, ep3=0.2\n'
            'Rising total loss is expected\n'
            'during warmup phase.')
    ax_kl.text(0.97, 0.95, note, transform=ax_kl.transAxes,
               fontsize=7.5, va='top', ha='right', color='#555',
               bbox=dict(boxstyle='round', facecolor='#FFFDE7',
                         edgecolor='#F9A825', linewidth=1))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('#FAFAFA')

    fig.suptitle(
        'Diabetes VAE — Project Summary\n'
        'Latent Metabolic State Learning with Variational Autoencoders  ·  MetaboNet Dataset',
        fontsize=13, fontweight='bold', y=0.98, color='#212121'
    )

    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32,
                          left=0.04, right=0.97, top=0.93, bottom=0.04)

    ax_pipe   = fig.add_subplot(gs[0, :2])
    ax_src    = fig.add_subplot(gs[0,  2])
    ax_model  = fig.add_subplot(gs[1, :2])
    ax_anneal = fig.add_subplot(gs[1,  2])
    ax_loss   = fig.add_subplot(gs[2,  0])
    ax_kl     = fig.add_subplot(gs[2,  1])
    ax_notes  = fig.add_subplot(gs[2,  2])

    panel_pipeline(ax_pipe)
    panel_sources(ax_src)
    panel_model(ax_model)
    panel_annealing(ax_anneal)
    panel_history(ax_loss, ax_kl)

    # Panel G: key design decisions
    ax_notes.axis('off')
    ax_notes.set_title('G  |  Key Design Decisions', fontsize=11,
                       fontweight='bold', loc='left', pad=8)
    decisions = [
        ('Dataset', [
            '• 5-min grid → 288 slots/day, no resampling needed',
            '• ≥50 % CGM coverage filter (144 readings)',
            '• Linear interp of gaps ≤1 h; ffill/bfill for edges',
            '• Physiology: mean HR, sum steps, mean GSR/skin_temp/cal',
            '• Missing wearable day → zero vector (not dropped)',
        ]),
        ('Model', [
            '• Separate Conv1D encoders for CGM & insulin (stride=1)',
            '• MLP encoder for 5-dim physiology daily summary',
            '• Product-of-Experts fusion → principled missing data',
            '• Shared latent space: z ∈ ℝ⁸',
            '• 668 K parameters — CPU-trainable',
        ]),
        ('Training', [
            '• KL annealing: weight 0→1 over first 10 epochs',
            '• Gradient clipping: max_norm = 1.0',
            '• logvar clamped to [–10, 10] (prevents exp overflow)',
            '• lr = 3×10⁻⁴, Adam, weight decay 10⁻⁵',
            '• Checkpoint every 5 epochs + best_model.pt',
        ]),
    ]
    y = 0.95
    for section, points in decisions:
        ax_notes.text(0.02, y, section, fontsize=9, fontweight='bold',
                      transform=ax_notes.transAxes, color='#333')
        y -= 0.07
        for pt in points:
            ax_notes.text(0.04, y, pt, fontsize=7.8,
                          transform=ax_notes.transAxes, color='#444')
            y -= 0.06
        y -= 0.03

    out = Path('results/project_summary.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved to {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
