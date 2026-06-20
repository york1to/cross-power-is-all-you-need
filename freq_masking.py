#!/usr/bin/env python3
"""
Experiment: Frequency Band Masking Intervention
================================================
Addresses reviewer comment: "Intervention experiments (e.g., frequency band
masking) would provide stronger evidence for magnitude-aware weighting."

Gradient attribution shows correlation between learned weighting and |G12|,
but correlation != causation. This experiment establishes causation:

  For each frequency bin k:
    1. Zero out bin k in the input tokens
    2. Measure MAE change: Δ_MAE[k] = MAE(masked) - MAE(full)
    3. Compare Δ_MAE profile with |G12| profile

If the network truly weights by magnitude:
  - Masking high-|G12| bins should increase MAE substantially
  - Masking low-|G12| bins should have minimal effect
  - cor(Δ_MAE, |G12|) should be positive

As controls, we also compare with:
  - PHAT weighting 1/|G12|: should show negative or zero correlation
  - Gradient attribution w[k]: should correlate positively with Δ_MAE
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")
os.makedirs('results', exist_ok=True)


# ─────────────────────────────────────────────────────────────
# DATA (self-contained, no perfreq_probe import)
# ─────────────────────────────────────────────────────────────

def colored_noise(T, beta=1.0, rng=None):
    rng = rng or np.random
    f = np.fft.rfftfreq(T); f[0] = 1.0
    pwr = f ** (-beta / 2); pwr[0] = 0.0
    return np.fft.irfft(
        (rng.randn(len(f)) + 1j*rng.randn(len(f))) * pwr, T).astype(np.float32)

def make_pair(tau, snr_db, T=256, noise_type='white', rng=None):
    rng = rng or np.random
    src = (rng.randn(T) if noise_type == 'white'
           else colored_noise(T, 1.0, rng)).astype(np.float32)
    src /= np.std(src) + 1e-8
    S = np.fft.rfft(src)
    k = np.arange(len(S))
    mic2_clean = np.fft.irfft(S * np.exp(-1j * 2*np.pi*k*tau/T), T).astype(np.float32)
    if np.isinf(snr_db):
        return src.copy(), mic2_clean
    ns = 10**(-snr_db / 20)
    return (src + rng.randn(T).astype(np.float32)*ns,
            mic2_clean + rng.randn(T).astype(np.float32)*ns)

def to_tokens(mic1, mic2):
    X1, X2 = np.fft.rfft(mic1), np.fft.rfft(mic2)
    t = np.stack([X1.real, X1.imag, X2.real, X2.imag], axis=1).astype(np.float32)
    return t / (np.std(t) + 1e-8)

class TDOADataset(Dataset):
    def __init__(self, N, tau_max=30, snr_db=0., T=256, noise_type='white', seed=0):
        rng = np.random.RandomState(seed)
        taus = rng.uniform(-tau_max, tau_max, N).astype(np.float32)
        toks = np.stack([to_tokens(*make_pair(t, snr_db, T, noise_type, rng))
                         for t in tqdm(taus, desc='  data', leave=False)])
        self.tokens    = torch.from_numpy(toks)
        self.taus_norm = torch.from_numpy(taus / tau_max)
        self.taus_raw  = torch.from_numpy(taus)
        self.T, self.F, self.tau_max = T, T//2+1, tau_max
    def __len__(self): return len(self.taus_norm)
    def __getitem__(self, i): return self.tokens[i], self.taus_norm[i:i+1]


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

    def forward(self, x):
        B = x.shape[0]
        h = self.input_proj(x) + self.freq_embed(torch.arange(self.F, device=x.device))
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        for layer in self.layers_:
            h = layer(h)
        return self.head(self.norm_out(h[:, 0]))

class CNNFreqModel(nn.Module):
    def __init__(self, F, d=64, n_layers=4, kernel=5):
        super().__init__()
        self.F, self.d, self.n_layers = F, d, n_layers
        pad = kernel // 2
        self.layers_ = nn.ModuleList()
        in_ch = 4
        for i in range(n_layers):
            self.layers_.append(nn.Sequential(
                nn.Conv1d(in_ch, d, kernel, padding=pad),
                nn.LayerNorm([d, F]),
                nn.GELU()
            ))
            in_ch = d
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = x.permute(0, 2, 1)
        for layer in self.layers_:
            h = layer(h)
        return self.head(h.mean(dim=-1))


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
        pbar.set_postfix({'tr': f'{tm:.3f}', 'va': f'{vm:.3f}'})

# ── Config ──
T, TAU_MAX = 256, 30
N_TRAIN, N_VAL = 50_000, 5_000
EPOCHS = 120
N_EVAL = 2000


def compute_baseline_mae(model, dataset, idx):
    """Compute MAE on unmasked inputs."""
    model.eval()
    toks = dataset.tokens[idx].to(device)
    taus = dataset.taus_raw[idx].numpy()
    BS = 256

    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), BS):
            pred = model(toks[s:s+BS]).cpu().numpy().flatten()
            preds.append(pred)
    preds = np.concatenate(preds) * dataset.tau_max
    return np.abs(preds - taus)  # per-sample AE, shape (N,)


def compute_single_bin_masking(model, dataset, idx):
    """Mask each frequency bin individually, measure MAE change."""
    model.eval()
    F = dataset.F
    toks = dataset.tokens[idx].to(device)
    taus = dataset.taus_raw[idx].numpy()
    BS = 256
    N = len(idx)

    # Baseline predictions
    baseline_preds = []
    with torch.no_grad():
        for s in range(0, N, BS):
            pred = model(toks[s:s+BS]).cpu().numpy().flatten()
            baseline_preds.append(pred)
    baseline_preds = np.concatenate(baseline_preds) * dataset.tau_max
    baseline_ae = np.abs(baseline_preds - taus)
    baseline_mae = baseline_ae.mean()

    # Per-bin masking
    delta_mae = np.zeros(F)
    for k in tqdm(range(F), desc='  masking bins'):
        masked_preds = []
        with torch.no_grad():
            for s in range(0, N, BS):
                batch = toks[s:s+BS].clone()
                batch[:, k, :] = 0.0  # zero out bin k
                pred = model(batch).cpu().numpy().flatten()
                masked_preds.append(pred)
        masked_preds = np.concatenate(masked_preds) * dataset.tau_max
        masked_mae = np.abs(masked_preds - taus).mean()
        delta_mae[k] = masked_mae - baseline_mae

    return delta_mae, baseline_mae


def compute_band_masking(model, dataset, idx, band_width=10):
    """Mask contiguous frequency bands, measure MAE change."""
    model.eval()
    F = dataset.F
    toks = dataset.tokens[idx].to(device)
    taus = dataset.taus_raw[idx].numpy()
    BS = 256
    N = len(idx)

    # Baseline
    baseline_preds = []
    with torch.no_grad():
        for s in range(0, N, BS):
            pred = model(toks[s:s+BS]).cpu().numpy().flatten()
            baseline_preds.append(pred)
    baseline_preds = np.concatenate(baseline_preds) * dataset.tau_max
    baseline_mae = np.abs(baseline_preds - taus).mean()

    # Band masking
    n_bands = F // band_width
    band_delta = np.zeros(n_bands)
    band_centers = np.zeros(n_bands)
    for b in range(n_bands):
        k_start = b * band_width
        k_end = min(k_start + band_width, F)
        band_centers[b] = (k_start + k_end) / 2

        masked_preds = []
        with torch.no_grad():
            for s in range(0, N, BS):
                batch = toks[s:s+BS].clone()
                batch[:, k_start:k_end, :] = 0.0
                pred = model(batch).cpu().numpy().flatten()
                masked_preds.append(pred)
        masked_preds = np.concatenate(masked_preds) * dataset.tau_max
        masked_mae = np.abs(masked_preds - taus).mean()
        band_delta[b] = masked_mae - baseline_mae

    return band_delta, band_centers, baseline_mae


def compute_magnitude_profile(dataset, idx):
    """Compute average |G12| profile."""
    toks = dataset.tokens[idx].numpy()
    X1 = toks[:, :, 0] + 1j * toks[:, :, 1]
    X2 = toks[:, :, 2] + 1j * toks[:, :, 3]
    cross = X1 * np.conj(X2)
    return np.abs(cross).mean(axis=0)  # (F,)


def compute_gradient_attribution(model, dataset, idx, n_samples=2000):
    """Compute gradient-based frequency attribution."""
    model.eval()
    BS = 64
    all_grad_w = []

    for s in tqdm(range(0, min(len(idx), n_samples), BS), desc='  grads', leave=False):
        batch_idx = idx[s:s+BS]
        toks = dataset.tokens[batch_idx].to(device).requires_grad_(True)
        pred = model(toks)
        pred.sum().backward()
        grad = toks.grad.detach()
        grad_w = grad.norm(dim=-1)  # (B, F)
        all_grad_w.append(grad_w.cpu().numpy())
        model.zero_grad()

    return np.concatenate(all_grad_w).mean(axis=0)  # (F,)


def plot_masking_results(delta_mae, mag_profile, grad_w, arch, condition, path):
    """Plot masking sensitivity vs magnitude and gradient profiles."""
    F = len(delta_mae)
    freqs = np.arange(F)

    # Normalize all profiles to [0, 1] for visual comparison
    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: Raw masking sensitivity
    ax = axes[0, 0]
    ax.bar(freqs, delta_mae, color='steelblue', alpha=0.7, width=1.0)
    ax.set_xlabel('Frequency bin k')
    ax.set_ylabel('ΔMAE (samples)')
    ax.set_title('Single-bin masking sensitivity', fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel B: Normalized overlay
    ax = axes[0, 1]
    ax.plot(freqs, norm01(delta_mae), '-', color='steelblue', lw=2,
            label='Masking ΔMAE')
    ax.plot(freqs, norm01(mag_profile), '--', color='gray', lw=2,
            label='|G₁₂| magnitude')
    ax.plot(freqs, norm01(grad_w), ':', color='darkorange', lw=2,
            label='Gradient attribution')
    phat_w = 1.0 / (mag_profile + 1e-10)
    ax.plot(freqs, norm01(phat_w), '-.', color='royalblue', lw=1.5,
            alpha=0.7, label='PHAT 1/|G₁₂|')
    ax.set_xlabel('Frequency bin k')
    ax.set_ylabel('Normalized value')
    ax.set_title('Profile comparison (normalized)', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel C: Scatter — ΔMAE vs |G12|
    ax = axes[1, 0]
    r_mag, p_mag = pearsonr(delta_mae[1:], mag_profile[1:])  # skip DC
    ax.scatter(mag_profile[1:], delta_mae[1:], s=15, alpha=0.6, color='gray')
    ax.set_xlabel('|G₁₂|[k] (magnitude)')
    ax.set_ylabel('ΔMAE[k]')
    ax.set_title(f'ΔMAE vs Magnitude (r={r_mag:.3f}, p={p_mag:.1e})',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel D: Scatter — ΔMAE vs gradient attribution
    ax = axes[1, 1]
    r_grad, p_grad = pearsonr(delta_mae[1:], grad_w[1:])
    ax.scatter(grad_w[1:], delta_mae[1:], s=15, alpha=0.6, color='darkorange')
    ax.set_xlabel('Gradient attribution w[k]')
    ax.set_ylabel('ΔMAE[k]')
    ax.set_title(f'ΔMAE vs Gradient (r={r_grad:.3f}, p={p_grad:.1e})',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    plt.suptitle(f'Frequency Masking Intervention ({arch}, {condition})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return r_mag, r_grad


# ── Main ──
if __name__ == '__main__':
    CONDITIONS = [
        dict(noise_type='colored', snr_db=0.0),
        dict(noise_type='white',   snr_db=0.0),
    ]
    ARCHS = [
        ('transformer', lambda F: TransformerModel(F, d_model=64, n_layers=4)),
        ('cnn',         lambda F: CNNFreqModel(F, d=64, n_layers=4, kernel=5)),
    ]

    all_results = {}

    for cfg in CONDITIONS:
        tag = f"{cfg['noise_type']}_snr{cfg['snr_db']:+.0f}dB"
        condition = f"{cfg['noise_type']}, {cfg['snr_db']:+.0f} dB"
        print(f"\n{'='*60}")
        print(f"  Condition: {condition}")
        print(f"{'='*60}")

        tr_ds = TDOADataset(N_TRAIN, TAU_MAX, cfg['snr_db'], T,
                            cfg['noise_type'], seed=42)
        va_ds = TDOADataset(N_VAL, TAU_MAX, cfg['snr_db'], T,
                            cfg['noise_type'], seed=99)

        for arch_name, arch_fn in ARCHS:
            print(f"\n  --- {arch_name} ---")

            model = arch_fn(tr_ds.F).to(device)
            print(f"  Training ({sum(p.numel() for p in model.parameters()):,} params)...")
            train_model(model, tr_ds, va_ds, epochs=EPOCHS)

            rng = np.random.RandomState(777)
            idx = rng.choice(len(va_ds), N_EVAL, replace=False)

            # 1. Magnitude profile
            print("  Computing magnitude profile...")
            mag_profile = compute_magnitude_profile(va_ds, idx)

            # 2. Gradient attribution
            print("  Computing gradient attribution...")
            grad_w = compute_gradient_attribution(model, va_ds, idx)

            # 3. Single-bin masking
            print("  Running single-bin masking (129 bins)...")
            delta_mae, baseline_mae = compute_single_bin_masking(
                model, va_ds, idx)
            print(f"  Baseline MAE: {baseline_mae:.3f}")

            # 4. Correlations (skip DC)
            r_mag, p_mag = pearsonr(delta_mae[1:], mag_profile[1:])
            phat_w = 1.0 / (mag_profile[1:] + 1e-10)
            r_phat, p_phat = pearsonr(delta_mae[1:], phat_w)
            r_grad, p_grad = pearsonr(delta_mae[1:], grad_w[1:])

            print(f"\n  Correlations (ΔMAE vs):")
            print(f"    |G12| magnitude:    r={r_mag:+.3f}  (p={p_mag:.1e})")
            print(f"    PHAT 1/|G12|:       r={r_phat:+.3f}  (p={p_phat:.1e})")
            print(f"    Gradient attrib:    r={r_grad:+.3f}  (p={p_grad:.1e})")

            # 5. Band masking (coarser, for robustness check)
            print("  Running band masking (width=10)...")
            band_delta, band_centers, _ = compute_band_masking(
                model, va_ds, idx, band_width=10)

            # 6. Plot
            key = f"{tag}_{arch_name}"
            r_mag_plot, r_grad_plot = plot_masking_results(
                delta_mae, mag_profile, grad_w, arch_name, condition,
                f'results/freq_masking_{key}.png')

            all_results[key] = {
                'delta_mae': delta_mae,
                'mag_profile': mag_profile,
                'grad_w': grad_w,
                'baseline_mae': baseline_mae,
                'r_mag': r_mag, 'p_mag': p_mag,
                'r_phat': r_phat, 'p_phat': p_phat,
                'r_grad': r_grad, 'p_grad': p_grad,
                'band_delta': band_delta,
                'band_centers': band_centers,
            }

    # Save
    torch.save(all_results, 'results/freq_masking.pt')

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Condition':30s}  {'r(ΔMAE,|G12|)':>14s}  {'r(ΔMAE,PHAT)':>13s}  "
          f"{'r(ΔMAE,grad)':>13s}")
    for key, res in all_results.items():
        print(f"  {key:30s}  {res['r_mag']:>+14.3f}  {res['r_phat']:>+13.3f}  "
              f"{res['r_grad']:>+13.3f}")
    print(f"\n  Saved: results/freq_masking.pt")
