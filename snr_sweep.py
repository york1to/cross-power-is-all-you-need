#!/usr/bin/env python3
"""
SNR Sweep — Phase Transition in Aggregation Strategy
=====================================================
Key hypothesis (à la Allen-Zhu):
  At high SNR, PHAT whitening IS theoretically optimal (for white noise).
  So the network might converge CLOSER to PHAT at high SNR.
  At low SNR, PHAT amplifies noise → network should DEVIATE from PHAT.

If we see phat_cos R² rising at high SNR and falling at low SNR,
that's a phase transition: the network transitions from near-PHAT
to non-PHAT behavior as noise increases.

Cross-power (cross_re) should remain high regardless of SNR —
it's the physics-forced invariant.

Sweep: SNR ∈ {+20, +10, +5, 0, -5, -10} dB
       noise ∈ {white, colored}
       architecture: Transformer (+ CNN for comparison)
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")
os.makedirs('results', exist_ok=True)

from perfreq_probe import (colored_noise, make_pair, to_tokens,
                            TDOADataset, build_probe_targets)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

T, TAU_MAX   = 256, 30
N_TRAIN, N_VAL = 50_000, 5_000
EPOCHS       = 120
F            = T // 2 + 1            # 129

SNR_LEVELS   = [20, 10, 5, 0, -5, -10]
NOISE_TYPES  = ['white', 'colored']


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

class TransformerModel(nn.Module):
    def __init__(self, F, d_model=64, n_layers=4, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.F, self.d_model, self.n_layers = F, d_model, n_layers
        self.input_proj = nn.Linear(4, d_model)
        self.freq_embed = nn.Embedding(F, d_model)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers_    = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, 1)

    def _embed(self, x):
        B = x.shape[0]
        h = self.input_proj(x)
        h = h + self.freq_embed(torch.arange(self.F, device=x.device))
        return torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)

    def forward(self, x, return_hidden=False, include_embed=True):
        h = self._embed(x)
        hiddens = [h.detach()] if (return_hidden and include_embed) else []
        for layer in self.layers_:
            h = layer(h)
            if return_hidden:
                hiddens.append(h.detach())
        out = self.head(self.norm_out(h[:, 0]))
        return (out, hiddens) if return_hidden else out

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNFreqModel(nn.Module):
    def __init__(self, F, d=64, n_layers=4, kernel=5):
        super().__init__()
        self.F, self.d, self.n_layers = F, d, n_layers
        pad = kernel // 2
        self.layers_ = nn.ModuleList()
        in_ch = 4
        for _ in range(n_layers):
            self.layers_.append(nn.Sequential(
                nn.Conv1d(in_ch, d, kernel, padding=pad),
                nn.LayerNorm([d, F]),
                nn.GELU()
            ))
            in_ch = d
        self.head = nn.Linear(d, 1)

    def forward(self, x, return_hidden=False):
        h = x.permute(0, 2, 1)
        hiddens = []
        for layer in self.layers_:
            h = layer(h)
            if return_hidden:
                hiddens.append(h.permute(0, 2, 1).detach())
        out = self.head(h.mean(dim=-1))
        return (out, hiddens) if return_hidden else out

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_model(model, train_ds, val_ds, epochs=120, lr=3e-4,
                batch_size=1024, warmup_frac=0.05):
    model.to(device)
    tr = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    va = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    total_steps  = epochs * len(tr)
    warmup_steps = int(warmup_frac * total_steps)

    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * p))

    sch     = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.HuberLoss()
    history = {'train_mae': [], 'val_mae': []}
    pbar = tqdm(range(epochs), desc='  train')
    for _ in pbar:
        model.train()
        tr_mae = []
        for toks, tau in tr:
            toks, tau = toks.to(device), tau.to(device)
            pred = model(toks)
            loss = loss_fn(pred, tau)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            tr_mae.append((pred - tau).abs().mean().item() * val_ds.tau_max)
        model.eval()
        va_mae = []
        with torch.no_grad():
            for toks, tau in va:
                toks, tau = toks.to(device), tau.to(device)
                va_mae.append((model(toks) - tau).abs().mean().item() * val_ds.tau_max)
        tm, vm = np.mean(tr_mae), np.mean(va_mae)
        history['train_mae'].append(tm); history['val_mae'].append(vm)
        pbar.set_postfix({'tr': f'{tm:.3f}', 'va': f'{vm:.3f}'})
    return history


# ─────────────────────────────────────────────────────────────
# PROBING
# ─────────────────────────────────────────────────────────────

def collect_and_probe(model, val_ds, arch, n_probe=3000, n_freq_sample=64):
    """Collect hidden states and run linear probes. Returns summary dict."""
    model.eval().to(device)
    rng = np.random.RandomState(7)
    idx = rng.choice(len(val_ds), n_probe, replace=False)
    targets = build_probe_targets(val_ds, idx)
    freq_bins = rng.choice(range(1, val_ds.F),
                           min(n_freq_sample, val_ds.F - 1), replace=False)
    split = int(0.7 * n_probe)
    toks_all = val_ds.tokens[idx].to(device)

    BS = 256
    if arch == 'transformer':
        n_depths = model.n_layers + 1
        f_buf = [[] for _ in range(n_depths)]
        g_buf = [[] for _ in range(n_depths)]
        with torch.no_grad():
            for s in range(0, n_probe, BS):
                _, hs = model(toks_all[s:s+BS], return_hidden=True, include_embed=True)
                for l, h in enumerate(hs):
                    f_buf[l].append(h[:, 1:].cpu().numpy())
                    g_buf[l].append(h[:, 0].cpu().numpy())
        freq_H   = [np.concatenate(b) for b in f_buf]
        global_H = [np.concatenate(b) for b in g_buf]

    elif arch == 'cnn':
        n_depths = model.n_layers
        f_buf = [[] for _ in range(n_depths)]
        with torch.no_grad():
            for s in range(0, n_probe, BS):
                _, hs = model(toks_all[s:s+BS], return_hidden=True)
                for l, h in enumerate(hs):
                    f_buf[l].append(h.cpu().numpy())
        freq_H   = [np.concatenate(b) for b in f_buf]
        global_H = [h.mean(axis=1) for h in freq_H]

    # Per-frequency probes
    freq_results = {}
    for name in ['cross_re', 'cross_im', 'phat_cos', 'phat_sin']:
        y = targets[name]
        r2s = []
        for l in range(n_depths):
            r2_per_k = []
            for k in freq_bins:
                H_tr = freq_H[l][:split, k, :]
                H_te = freq_H[l][split:, k, :]
                y_tr, y_te = y[:split, k], y[split:, k]
                reg = Ridge(alpha=1.0).fit(H_tr, y_tr)
                r2_per_k.append(max(r2_score(y_te, reg.predict(H_te)), -0.1))
            r2s.append(float(np.mean(r2_per_k)))
        freq_results[name] = r2s

    # Global probes
    global_results = {}
    for name, y in targets.items():
        r2s = []
        for l in range(n_depths):
            H_tr, H_te = global_H[l][:split], global_H[l][split:]
            y_tr, y_te = y[:split], y[split:]
            reg = Ridge(alpha=1.0).fit(H_tr, y_tr)
            y_hat = reg.predict(H_te)
            if y.ndim == 1:
                r2 = r2_score(y_te, y_hat)
            else:
                r2 = float(np.mean([max(r2_score(y_te[:, k], y_hat[:, k]), -0.1)
                                    for k in range(1, y.shape[1])]))
            r2s.append(r2)
        global_results[name] = r2s

    return freq_results, global_results, n_depths


# ─────────────────────────────────────────────────────────────
# MAIN SWEEP
# ─────────────────────────────────────────────────────────────

def run_sweep(arch_name, arch_factory):
    """Run full SNR sweep for one architecture across both noise types."""
    all_results = {}

    for noise_type in NOISE_TYPES:
        all_results[noise_type] = {}

        for snr_db in SNR_LEVELS:
            label = f"{noise_type}_snr{snr_db:+d}dB"
            print(f"\n{'─'*50}")
            print(f"  {arch_name} | {label}")
            print(f"{'─'*50}")

            print("  Data ...")
            tr_ds = TDOADataset(N_TRAIN, TAU_MAX, float(snr_db), T, noise_type, seed=42)
            va_ds = TDOADataset(N_VAL,   TAU_MAX, float(snr_db), T, noise_type, seed=99)

            print("  Train ...")
            model = arch_factory()
            t0 = time.time()
            hist = train_model(model, tr_ds, va_ds, epochs=EPOCHS, lr=3e-4, batch_size=1024)
            val_mae = hist['val_mae'][-1]
            print(f"  {time.time()-t0:.1f}s | val MAE: {val_mae:.3f}")

            print("  Probe ...")
            freq_p, global_p, n_depths = collect_and_probe(model, va_ds, arch_name)

            all_results[noise_type][snr_db] = {
                'freq': freq_p, 'global': global_p,
                'n_depths': n_depths, 'val_mae': val_mae,
            }

            # Key metrics
            cross_re_d1 = freq_p['cross_re'][1] if n_depths > 1 else freq_p['cross_re'][0]
            phat_cos_max = max(freq_p['phat_cos'])
            tau_last = global_p['tau'][-1]
            print(f"  cross_re@D1={cross_re_d1:.3f}  phat_cos_max={phat_cos_max:.3f}  "
                  f"tau@last={tau_last:.3f}  val_MAE={val_mae:.3f}")

    return all_results


def plot_phase_transition(results, arch_name):
    """
    Plot the phase transition: R² vs SNR for each noise type.
    3 panels per noise type:
      - cross_re at each depth vs SNR
      - phat_cos at each depth vs SNR
      - tau + val_MAE vs SNR
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, noise_type in enumerate(NOISE_TYPES):
        noise_res = results[noise_type]
        snrs = sorted(noise_res.keys())
        n_depths = noise_res[snrs[0]]['n_depths']

        depth_colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_depths))
        if arch_name == 'transformer':
            depth_labels = ['Emb'] + [f'L{i+1}' for i in range(n_depths - 1)]
        else:
            depth_labels = [f'D{i}' for i in range(n_depths)]

        # Panel 1: cross_re vs SNR at each depth
        ax = axes[row, 0]
        for d in range(n_depths):
            vals = [noise_res[s]['freq']['cross_re'][d] for s in snrs]
            ax.plot(snrs, vals, 'o-', color=depth_colors[d], lw=2, ms=7,
                    label=depth_labels[d])
        ax.set_title(f'{noise_type} noise — cross_re per depth',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('SNR (dB)'); ax.set_ylabel('R²')
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.axhline(0, c='gray', lw=0.8, ls=':')

        # Panel 2: phat_cos vs SNR at each depth
        ax = axes[row, 1]
        for d in range(n_depths):
            vals = [noise_res[s]['freq']['phat_cos'][d] for s in snrs]
            ax.plot(snrs, vals, 'o-', color=depth_colors[d], lw=2, ms=7,
                    label=depth_labels[d])
        ax.set_title(f'{noise_type} noise — phat_cos per depth\n(KEY: does high-SNR → higher PHAT?)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('SNR (dB)'); ax.set_ylabel('R²')
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.axhline(0, c='gray', lw=0.8, ls=':')
        ax.axhspan(-0.15, 0.3, alpha=0.06, color='red')

        # Panel 3: summary metrics vs SNR
        ax = axes[row, 2]
        cross_re_d1 = [noise_res[s]['freq']['cross_re'][min(1, n_depths-1)] for s in snrs]
        phat_max    = [max(noise_res[s]['freq']['phat_cos']) for s in snrs]
        tau_last    = [noise_res[s]['global']['tau'][-1] for s in snrs]
        val_maes    = [noise_res[s]['val_mae'] for s in snrs]

        ax.plot(snrs, cross_re_d1, 'o-', color='darkorange', lw=2.5, ms=9,
                label='cross_re @ D1')
        ax.plot(snrs, phat_max, 's-', color='royalblue', lw=2.5, ms=9,
                label='phat_cos (best depth)')
        ax.plot(snrs, tau_last, '^-', color='black', lw=2.5, ms=9,
                label='tau @ last depth')

        ax2 = ax.twinx()
        ax2.plot(snrs, val_maes, 'D--', color='gray', lw=1.5, ms=7, alpha=0.6,
                 label='val MAE (samples)')
        ax2.set_ylabel('Val MAE (samples)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        ax.set_title(f'{noise_type} noise — Summary',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('SNR (dB)'); ax.set_ylabel('R²')
        ax.set_ylim(-0.15, 1.05); ax.grid(alpha=0.3)
        ax.axhline(0, c='gray', lw=0.8, ls=':')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower right')

    plt.suptitle(
        f'SNR Sweep — Phase Transition in Aggregation Strategy ({arch_name})\n'
        f'Does high-SNR → network converges to PHAT? Does low-SNR → network deviates?',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = f'results/snr_sweep_{arch_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {path}")
    plt.close()


def plot_phase_transition_compact(results_by_arch):
    """
    Compact 2-panel figure: the 'money plot' for the phase transition.
    Left: white noise, Right: colored noise
    Each panel: cross_re@D1, phat_cos(best), tau@last vs SNR
    Overlaid for both architectures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    arch_styles = {
        'transformer': {'ls': '-',  'marker': 'o', 'lbl': 'Transformer'},
        'cnn':         {'ls': '--', 'marker': 's', 'lbl': 'CNN'},
    }
    metric_colors = {
        'cross_re': 'darkorange',
        'phat_cos': 'royalblue',
        'tau':      'black',
    }

    for col, noise_type in enumerate(NOISE_TYPES):
        ax = axes[col]

        for arch_name, results in results_by_arch.items():
            sty = arch_styles[arch_name]
            noise_res = results[noise_type]
            snrs = sorted(noise_res.keys())
            n_depths = noise_res[snrs[0]]['n_depths']

            cross_re_d1 = [noise_res[s]['freq']['cross_re'][min(1, n_depths-1)] for s in snrs]
            phat_max    = [max(noise_res[s]['freq']['phat_cos']) for s in snrs]
            tau_last    = [noise_res[s]['global']['tau'][-1] for s in snrs]

            ax.plot(snrs, cross_re_d1, marker=sty['marker'], ls=sty['ls'],
                    color=metric_colors['cross_re'], lw=2.5, ms=9, alpha=0.8)
            ax.plot(snrs, phat_max, marker=sty['marker'], ls=sty['ls'],
                    color=metric_colors['phat_cos'], lw=2.5, ms=9, alpha=0.8)
            ax.plot(snrs, tau_last, marker=sty['marker'], ls=sty['ls'],
                    color=metric_colors['tau'], lw=2.5, ms=9, alpha=0.8)

        ax.set_title(f'{noise_type.title()} Noise', fontsize=12, fontweight='bold')
        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel('R²', fontsize=11)
        ax.set_ylim(-0.15, 1.05)
        ax.grid(alpha=0.3)
        ax.axhline(0, c='gray', lw=0.8, ls=':')
        ax.axhspan(-0.15, 0.3, alpha=0.06, color='red')

    # Custom legend
    metric_handles = [
        Line2D([0], [0], color=metric_colors['cross_re'], lw=2.5, label='cross_re @ D1'),
        Line2D([0], [0], color=metric_colors['phat_cos'], lw=2.5, label='phat_cos (best depth)'),
        Line2D([0], [0], color=metric_colors['tau'], lw=2.5, label='tau @ last depth'),
    ]
    arch_handles = [
        Line2D([0], [0], color='gray', ls=sty['ls'], marker=sty['marker'], lw=2,
               label=sty['lbl'])
        for _, sty in arch_styles.items()
        if _ in results_by_arch
    ]
    axes[0].legend(handles=metric_handles + arch_handles, fontsize=9, loc='lower right')

    plt.suptitle(
        'SNR Phase Transition — "Does the network approach PHAT at high SNR?"\n'
        'Prediction: cross_re stays high (universal), phat_cos stays low (PHAT always rejected)',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = 'results/snr_sweep_compact.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results_by_arch = {}

    # Transformer sweep
    print(f"\n{'━'*60}")
    print("  SNR SWEEP: Transformer")
    print(f"{'━'*60}")
    tf_results = run_sweep(
        'transformer',
        lambda: TransformerModel(F=F, d_model=64, n_layers=4, n_heads=4, d_ff=256))
    results_by_arch['transformer'] = tf_results
    torch.save(tf_results, 'results/snr_sweep_transformer.pt')
    plot_phase_transition(tf_results, 'transformer')

    # CNN sweep
    print(f"\n{'━'*60}")
    print("  SNR SWEEP: CNN")
    print(f"{'━'*60}")
    cnn_results = run_sweep(
        'cnn',
        lambda: CNNFreqModel(F=F, d=64, n_layers=4, kernel=5))
    results_by_arch['cnn'] = cnn_results
    torch.save(cnn_results, 'results/snr_sweep_cnn.pt')
    plot_phase_transition(cnn_results, 'cnn')

    # Compact comparison
    plot_phase_transition_compact(results_by_arch)

    # Summary table
    print(f"\n{'━'*70}")
    print("SNR SWEEP SUMMARY")
    print(f"{'━'*70}")
    for arch_name, results in results_by_arch.items():
        print(f"\n  {arch_name.upper()}")
        print(f"  {'Noise':8s}  {'SNR':>5s}  {'val MAE':>8s}  {'cross_re@D1':>12s}  "
              f"{'phat_cos_max':>13s}  {'tau@last':>9s}")
        for noise_type in NOISE_TYPES:
            for snr_db in SNR_LEVELS:
                res = results[noise_type][snr_db]
                n = res['n_depths']
                cr = res['freq']['cross_re'][min(1, n-1)]
                pc = max(res['freq']['phat_cos'])
                tau = res['global']['tau'][-1]
                print(f"  {noise_type:8s}  {snr_db:>+5d}  {res['val_mae']:>8.3f}  "
                      f"{cr:>12.3f}  {pc:>13.3f}  {tau:>9.3f}")

    print(f"\nFigures: results/snr_sweep_*.png")
    print(f"Data:    results/snr_sweep_*.pt")
    print(f"\n{'━'*70}")
    print("DONE")
    print(f"{'━'*70}")
