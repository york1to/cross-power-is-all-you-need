#!/usr/bin/env python3
"""
Learned Weighting Analysis — "If not PHAT, then what?"
======================================================
Key question: The network rejects PHAT whitening. What weighting does it use instead?

Method: Gradient-based frequency attribution.
  For each sample, compute ∂τ_pred / ∂input_tokens → per-frequency sensitivity.
  The L2 norm of the gradient at each frequency bin k = "effective weight" w[k].

Compare with classical GCC weightings:
  PHAT:      W[k] = 1 / |X1[k]·X2*[k]|     (suppress magnitude, keep phase)
  Magnitude: W[k] = |X1[k]·X2*[k]|           (emphasize strong bins)
  SCOT:      W[k] = 1 / sqrt(|X1|²·|X2|²)   (coherence-based)
  Flat:      W[k] = 1                         (unweighted GCC)

If the learned weighting correlates with magnitude (opposite of PHAT),
it confirms the network prefers magnitude-aware aggregation.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
F            = T // 2 + 1


# ─────────────────────────────────────────────────────────────
# MODELS (same as architectures.py)
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
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
# GRADIENT-BASED FREQUENCY WEIGHTING
# ─────────────────────────────────────────────────────────────

def extract_gradient_weighting(model, dataset, n_samples=2000, batch_size=64):
    """
    Compute ∂τ_pred / ∂input_tokens for each sample.
    Returns per-frequency importance weights.
    """
    model.eval().to(device)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    all_grad_w = []    # gradient-based weight per freq
    all_cross_mag = [] # |X1·X2*| per freq
    all_auto1 = []     # |X1|² per freq
    all_auto2 = []     # |X2|² per freq

    for s in tqdm(range(0, len(idx), batch_size), desc='  grads', leave=False):
        batch_idx = idx[s:s + batch_size]
        toks = dataset.tokens[batch_idx].to(device).requires_grad_(True)

        pred = model(toks)
        pred.sum().backward()

        grad = toks.grad.detach()               # (B, F, 4)
        grad_w = grad.norm(dim=-1)              # (B, F)
        all_grad_w.append(grad_w.cpu().numpy())

        # Classical spectra from tokens
        toks_np = toks.detach().cpu().numpy()
        X1 = toks_np[:, :, 0] + 1j * toks_np[:, :, 1]
        X2 = toks_np[:, :, 2] + 1j * toks_np[:, :, 3]
        cross = X1 * np.conj(X2)
        all_cross_mag.append(np.abs(cross))
        all_auto1.append(np.abs(X1) ** 2)
        all_auto2.append(np.abs(X2) ** 2)

        model.zero_grad()

    grad_w    = np.concatenate(all_grad_w)       # (N, F)
    cross_mag = np.concatenate(all_cross_mag)    # (N, F)
    auto1     = np.concatenate(all_auto1)        # (N, F)
    auto2     = np.concatenate(all_auto2)        # (N, F)

    return grad_w, cross_mag, auto1, auto2


def compute_classical_weightings(cross_mag, auto1, auto2):
    """Compute classical GCC weighting functions."""
    eps = 1e-8
    phat = 1.0 / (cross_mag + eps)
    magnitude = cross_mag.copy()
    scot = 1.0 / (np.sqrt(auto1 * auto2) + eps)
    flat = np.ones_like(cross_mag)
    return {'PHAT': phat, 'Magnitude': magnitude, 'SCOT': scot, 'Flat': flat}


def normalize_profile(w):
    """Normalize weighting profile to sum to 1 for comparison."""
    s = w.sum(axis=-1, keepdims=True)
    return w / (s + 1e-8)


def correlation_analysis(learned, classical_dict, skip_dc=True):
    """Per-sample correlation between learned and each classical weighting."""
    start = 1 if skip_dc else 0
    results = {}
    for name, classical in classical_dict.items():
        cors = []
        for i in range(len(learned)):
            w_l = learned[i, start:]
            w_c = classical[i, start:]
            if np.std(w_l) < 1e-10 or np.std(w_c) < 1e-10:
                continue
            cor = np.corrcoef(w_l, w_c)[0, 1]
            cors.append(cor)
        results[name] = {
            'mean': float(np.mean(cors)),
            'std':  float(np.std(cors)),
        }
    return results


# ─────────────────────────────────────────────────────────────
# WEIGHT PROFILE PROBING
# ─────────────────────────────────────────────────────────────

def probe_effective_weights(model, val_ds, n_probe=3000, n_freq_sample=64):
    """
    For each frequency bin k, train a probe from the LAST layer global
    representation to predict the "contribution" of bin k to the output.

    Specifically: probe global_hidden → w[k] where w[k] is the gradient-based
    importance of frequency k. This tells us what the network "knows" about
    how it weights different frequencies.
    """
    model.eval().to(device)
    rng = np.random.RandomState(7)
    idx = rng.choice(len(val_ds), min(n_probe, len(val_ds)), replace=False)

    # Get gradient weights
    grad_w, cross_mag, _, _ = extract_gradient_weighting(
        model, val_ds, n_samples=n_probe, batch_size=64)

    # Get last-layer global representations (architecture-aware)
    toks_all = val_ds.tokens[idx].to(device)
    g_buf = []
    BS = 256
    is_transformer = isinstance(model, TransformerModel)
    with torch.no_grad():
        for s in range(0, n_probe, BS):
            batch = toks_all[s:s+BS]
            if is_transformer:
                _, hs = model(batch, return_hidden=True, include_embed=True)
                g_buf.append(hs[-1][:, 0].cpu().numpy())  # CLS at last layer
            else:
                _, hs = model(batch, return_hidden=True)
                g_buf.append(hs[-1].mean(dim=1).cpu().numpy())  # global avg pool
    global_H = np.concatenate(g_buf)  # (N, d)

    # Probe: global_hidden → gradient_weight[k] for each k
    freq_bins = rng.choice(range(1, val_ds.F),
                           min(n_freq_sample, val_ds.F - 1), replace=False)
    split = int(0.7 * n_probe)

    r2_per_k = []
    for k in freq_bins:
        H_tr, H_te = global_H[:split], global_H[split:]
        y_tr = grad_w[:split, k]
        y_te = grad_w[split:, k]
        reg = Ridge(alpha=1.0).fit(H_tr, y_tr)
        r2_per_k.append(max(r2_score(y_te, reg.predict(H_te)), -0.1))

    return float(np.mean(r2_per_k)), r2_per_k


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_weighting_analysis(results_by_arch):
    """
    For each architecture:
      - Panel 1: Average weighting profile across frequency
      - Panel 2: Correlation bar chart
    """
    archs = list(results_by_arch.keys())
    n = len(archs)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 10))
    if n == 1:
        axes = axes[:, None]

    arch_labels = {'transformer': 'Transformer', 'cnn': '1D-CNN'}
    classical_colors = {
        'PHAT': 'royalblue', 'Magnitude': 'darkorange',
        'SCOT': 'green', 'Flat': 'gray'
    }

    for col, arch in enumerate(archs):
        res = results_by_arch[arch]

        # Panel 1: Frequency profiles
        ax = axes[0, col]
        freqs = np.arange(1, F)  # skip DC
        # Normalize profiles for comparison
        learned_avg = res['learned_profile_avg'][1:]
        learned_avg = learned_avg / (learned_avg.sum() + 1e-8)

        ax.plot(freqs, learned_avg, 'k-', lw=2.5, label='Learned', zorder=5)
        for name in ['PHAT', 'Magnitude', 'SCOT']:
            profile = res['classical_profiles_avg'][name][1:]
            profile = profile / (profile.sum() + 1e-8)
            ax.plot(freqs, profile, '--', color=classical_colors[name],
                    lw=1.5, alpha=0.7, label=name)

        ax.set_title(f'{arch_labels.get(arch, arch)}\nAverage Frequency Weighting Profile',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency bin k')
        ax.set_ylabel('Normalized weight')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Panel 2: Correlation analysis
        ax = axes[1, col]
        cors = res['correlations']
        names = list(cors.keys())
        means = [cors[n]['mean'] for n in names]
        stds  = [cors[n]['std'] for n in names]
        colors = [classical_colors.get(n, 'gray') for n in names]

        bars = ax.bar(range(len(names)), means, yerr=stds,
                      color=colors, alpha=0.8, capsize=5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel('Pearson Correlation')
        ax.set_ylim(-0.5, 1.0)
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_title(f'{arch_labels.get(arch, arch)}\nCorrelation: Learned vs Classical',
                     fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        # Annotate
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.03, f'{m:.2f}', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle(
        'Learned Frequency Weighting — "If not PHAT, then what?"\n'
        'Gradient-based attribution: ||∂τ_pred/∂token[k]|| = effective weight at frequency k',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = 'results/learned_weighting.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {path}")
    plt.close()


def plot_snr_weighting_comparison(results_by_snr, arch_name):
    """How does the learned weighting change with SNR?"""
    snrs = sorted(results_by_snr.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    cmap = plt.cm.coolwarm
    norm_snr = plt.Normalize(vmin=min(snrs), vmax=max(snrs))

    # Panel 1: Learned profiles at each SNR
    ax = axes[0]
    freqs = np.arange(1, F)
    for snr in snrs:
        profile = results_by_snr[snr]['learned_profile_avg'][1:]
        profile = profile / (profile.sum() + 1e-8)
        ax.plot(freqs, profile, color=cmap(norm_snr(snr)), lw=2,
                label=f'SNR={snr:+d}dB')
    ax.set_title(f'{arch_name} — Learned Weighting vs SNR',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency bin k'); ax.set_ylabel('Normalized weight')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Correlation with classical weightings vs SNR
    ax = axes[1]
    for classical_name, color in [('PHAT', 'royalblue'), ('Magnitude', 'darkorange'),
                                   ('SCOT', 'green')]:
        cors = [results_by_snr[s]['correlations'][classical_name]['mean'] for s in snrs]
        ax.plot(snrs, cors, 'o-', color=color, lw=2.5, ms=9, label=classical_name)
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_title(f'{arch_name} — Correlation with Classical Weightings vs SNR',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Pearson Correlation')
    ax.set_ylim(-0.5, 1.0)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(
        'SNR-Dependent Weighting — Does the network adapt its strategy?',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = f'results/learned_weighting_snr_{arch_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Part 1: Main analysis at colored noise 0dB ──
    print(f"\n{'━'*60}")
    print("  PART 1: Learned Weighting Analysis (colored, SNR=0 dB)")
    print(f"{'━'*60}")

    CFG = dict(noise_type='colored', snr_db=0.0)
    tag = f"{CFG['noise_type']}_snr{CFG['snr_db']:+.0f}dB"

    print("  Data ...")
    tr_ds = TDOADataset(N_TRAIN, TAU_MAX, CFG['snr_db'], T, CFG['noise_type'], seed=42)
    va_ds = TDOADataset(N_VAL,   TAU_MAX, CFG['snr_db'], T, CFG['noise_type'], seed=99)

    results_by_arch = {}
    for arch_name, model in [
        ('transformer', TransformerModel(F=F, d_model=64, n_layers=4, n_heads=4, d_ff=256)),
        ('cnn',         CNNFreqModel(F=F, d=64, n_layers=4, kernel=5)),
    ]:
        print(f"\n  ── {arch_name} ({model.n_params:,} params) ──")

        print("     Training ...")
        hist = train_model(model, tr_ds, va_ds, epochs=EPOCHS, lr=3e-4, batch_size=1024)
        print(f"     val MAE: {hist['val_mae'][-1]:.3f}")

        print("     Extracting gradient weighting ...")
        grad_w, cross_mag, auto1, auto2 = extract_gradient_weighting(
            model, va_ds, n_samples=2000)

        classical = compute_classical_weightings(cross_mag, auto1, auto2)
        cors = correlation_analysis(grad_w, classical)

        print(f"     Correlations:")
        for name, c in cors.items():
            print(f"       {name:12s}: r={c['mean']:+.3f} ± {c['std']:.3f}")

        # Probe: can the network predict its own weighting?
        print("     Probing weight decodability ...")
        weight_r2, _ = probe_effective_weights(model, va_ds, n_probe=2000)
        print(f"     Weight decodability R²: {weight_r2:.3f}")

        results_by_arch[arch_name] = {
            'learned_profile_avg': grad_w.mean(axis=0),
            'classical_profiles_avg': {n: v.mean(axis=0) for n, v in classical.items()},
            'correlations': cors,
            'weight_decodability_r2': weight_r2,
            'val_mae': hist['val_mae'][-1],
        }

    torch.save(results_by_arch, f'results/learned_weighting_{tag}.pt')
    plot_weighting_analysis(results_by_arch)

    # ── Part 2: SNR-dependent weighting (Transformer only) ──
    print(f"\n{'━'*60}")
    print("  PART 2: SNR-Dependent Weighting (Transformer)")
    print(f"{'━'*60}")

    SNR_LEVELS = [20, 10, 5, 0, -5, -10]
    results_by_snr = {}

    for snr_db in SNR_LEVELS:
        print(f"\n  ── SNR={snr_db:+d} dB ──")
        tr_ds_snr = TDOADataset(N_TRAIN, TAU_MAX, float(snr_db), T, 'colored', seed=42)
        va_ds_snr = TDOADataset(N_VAL,   TAU_MAX, float(snr_db), T, 'colored', seed=99)

        model = TransformerModel(F=F, d_model=64, n_layers=4, n_heads=4, d_ff=256)
        hist = train_model(model, tr_ds_snr, va_ds_snr, epochs=EPOCHS, lr=3e-4, batch_size=1024)
        print(f"     val MAE: {hist['val_mae'][-1]:.3f}")

        grad_w, cross_mag, auto1, auto2 = extract_gradient_weighting(
            model, va_ds_snr, n_samples=2000)
        classical = compute_classical_weightings(cross_mag, auto1, auto2)
        cors = correlation_analysis(grad_w, classical)

        print(f"     Correlations: PHAT={cors['PHAT']['mean']:+.3f}, "
              f"Mag={cors['Magnitude']['mean']:+.3f}, SCOT={cors['SCOT']['mean']:+.3f}")

        results_by_snr[snr_db] = {
            'learned_profile_avg': grad_w.mean(axis=0),
            'classical_profiles_avg': {n: v.mean(axis=0) for n, v in classical.items()},
            'correlations': cors,
            'val_mae': hist['val_mae'][-1],
        }

    torch.save(results_by_snr, 'results/learned_weighting_snr_transformer.pt')
    plot_snr_weighting_comparison(results_by_snr, 'Transformer')

    # Summary
    print(f"\n{'━'*60}")
    print("LEARNED WEIGHTING SUMMARY")
    print(f"{'━'*60}")

    print("\n  Part 1: Architecture comparison (colored noise, 0 dB)")
    print(f"  {'Arch':15s}  {'cor(PHAT)':>10s}  {'cor(Mag)':>10s}  {'cor(SCOT)':>10s}  {'w_decode_R²':>12s}")
    for arch, res in results_by_arch.items():
        c = res['correlations']
        print(f"  {arch:15s}  {c['PHAT']['mean']:>+10.3f}  {c['Magnitude']['mean']:>+10.3f}  "
              f"{c['SCOT']['mean']:>+10.3f}  {res['weight_decodability_r2']:>12.3f}")

    print("\n  Part 2: SNR-dependent weighting (Transformer, colored noise)")
    print(f"  {'SNR':>5s}  {'cor(PHAT)':>10s}  {'cor(Mag)':>10s}  {'cor(SCOT)':>10s}  {'val MAE':>8s}")
    for snr, res in sorted(results_by_snr.items()):
        c = res['correlations']
        print(f"  {snr:>+5d}  {c['PHAT']['mean']:>+10.3f}  {c['Magnitude']['mean']:>+10.3f}  "
              f"{c['SCOT']['mean']:>+10.3f}  {res['val_mae']:>8.3f}")

    print(f"\nFigures: results/learned_weighting*.png")
    print(f"\n{'━'*60}")
    print("DONE")
    print(f"{'━'*60}")
