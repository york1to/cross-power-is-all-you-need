#!/usr/bin/env python3
"""
Per-Frequency Linear Probe Analysis for GCC-PHAT Mechanistic Interpretability
==============================================================================
Key question: does the Transformer re-discover GCC-PHAT computationally?

GCC-PHAT algorithm steps per frequency bin k:
  Input:    [Re(X1[k]), Im(X1[k]), Re(X2[k]), Im(X2[k])]
  Step 1:   cross_re[k] = Re(X1·X2*) = X1_re·X2_re + X1_im·X2_im   ← bilinear!
            cross_im[k] = Im(X1·X2*) = X1_im·X2_re - X1_re·X2_im
  Step 2:   phat_cos[k] = cos(2πkτ/T)  (after PHAT whitening → only phase remains)
  Step 3:   tau = TDOA   (global argmax / aggregation across all k)

If the Transformer re-discovers GCC-PHAT:
  Embed → L1: cross_re/im per-freq probe R² should JUMP
              (because cross_re is bilinear → not linearly decodable from input,
               but FFN in layer 1 can compute it as a ReLU network)
  L1 → L2:   phat_cos/sin per-freq probe R² should JUMP
              (whitening step: normalise cross by |cross|)
  L2 → L4:   CLS tau probe R² builds up   (global aggregation)

Key insight: CLS probe CANNOT detect step 1 — GCC cross-power is per-frequency,
             so only per-frequency token probing can find it.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")
os.makedirs('results', exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. DATA  (identical to experiment.py)
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
    S   = np.fft.rfft(src)
    k   = np.arange(len(S))
    mic2_clean = np.fft.irfft(S * np.exp(-1j * 2*np.pi*k*tau/T), T).astype(np.float32)
    if np.isinf(snr_db):
        return src.copy(), mic2_clean
    ns = 10**(-snr_db / 20)
    return (src        + rng.randn(T).astype(np.float32)*ns,
            mic2_clean + rng.randn(T).astype(np.float32)*ns)

def to_tokens(mic1, mic2):
    X1, X2 = np.fft.rfft(mic1), np.fft.rfft(mic2)
    t = np.stack([X1.real, X1.imag, X2.real, X2.imag], axis=1).astype(np.float32)
    return t / (np.std(t) + 1e-8)

class TDOADataset(Dataset):
    def __init__(self, N, tau_max=30, snr_db=0., T=256, noise_type='white', seed=0):
        rng  = np.random.RandomState(seed)
        taus = rng.uniform(-tau_max, tau_max, N).astype(np.float32)
        toks = np.stack([to_tokens(*make_pair(t, snr_db, T, noise_type, rng))
                         for t in tqdm(taus, desc='  data', leave=False)])
        self.tokens    = torch.from_numpy(toks)
        self.taus_norm = torch.from_numpy(taus / tau_max)
        self.taus_raw  = torch.from_numpy(taus)
        self.T, self.F, self.tau_max = T, T//2+1, tau_max

    def __len__(self): return len(self.taus_norm)
    def __getitem__(self, i):
        return self.tokens[i], self.taus_norm[i:i+1]


# ─────────────────────────────────────────────────────────────
# 2. MODEL  (extended: include_embed captures embedding layer)
# ─────────────────────────────────────────────────────────────

class TDOATransformer(nn.Module):
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

    def forward(self, x, return_hidden=False, include_embed=False):
        """
        include_embed=True: first hidden = output of embedding layer (before any
        transformer layer), so hiddens has length n_layers+1.
        """
        h = self._embed(x)
        hiddens = []
        if return_hidden and include_embed:
            hiddens.append(h.detach())          # 'Embed' checkpoint
        for layer in self.layers_:
            h = layer(h)
            if return_hidden:
                hiddens.append(h.detach())      # L1, L2, L3, L4
        out = self.head(self.norm_out(h[:, 0]))
        return (out, hiddens) if return_hidden else out

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# 3. TRAINING
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
        if step < warmup_steps:
            return step / max(1, warmup_steps)
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
# 4. PROBE TARGETS
# ─────────────────────────────────────────────────────────────

def build_probe_targets(val_ds, idx):
    """
    tau          : (N,)   — normalised TDOA
    cross_re/im  : (N, F) — Re/Im of X1·X2*   (GCC cross-power, bilinear in input)
    phat_cos/sin : (N, F) — cos/sin(2πkτ/T)   (PHAT-normalised phase per frequency)
    """
    T, F = val_ds.T, val_ds.F
    toks  = val_ds.tokens[idx].numpy()           # (N, F, 4)
    taus  = val_ds.taus_raw[idx].numpy()         # (N,)
    X1    = toks[:, :, 0] + 1j*toks[:, :, 1]   # (N, F)
    X2    = toks[:, :, 2] + 1j*toks[:, :, 3]
    cross = X1 * np.conj(X2)                     # (N, F)
    k     = np.arange(F)[None, :]               # (1, F)
    phi   = 2*np.pi * k * taus[:, None] / T     # (N, F)
    return {
        'tau':      (taus / val_ds.tau_max).astype(np.float32),   # (N,)
        'cross_re': cross.real.astype(np.float32),                 # (N, F)
        'cross_im': cross.imag.astype(np.float32),
        'phat_cos': np.cos(phi).astype(np.float32),
        'phat_sin': np.sin(phi).astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────
# 5a. CLS PROBE  (global CLS token → scalar or averaged target)
# ─────────────────────────────────────────────────────────────

def run_cls_probes(model, val_ds, n_probe=3000):
    """
    Probe CLS hidden state at each depth (Embed, L1…L4) → each target.
    For vector targets (cross, phat), R² is averaged over frequency bins.
    """
    model.eval().to(device)
    rng = np.random.RandomState(7)
    idx = rng.choice(len(val_ds), n_probe, replace=False)
    targets = build_probe_targets(val_ds, idx)

    toks_all = val_ds.tokens[idx].to(device)
    n_depths = model.n_layers + 1                  # Embed + 4 layers
    buf = [[] for _ in range(n_depths)]
    BS  = 256
    with torch.no_grad():
        for s in range(0, n_probe, BS):
            _, hs = model(toks_all[s:s+BS], return_hidden=True, include_embed=True)
            for l, h in enumerate(hs):
                buf[l].append(h[:, 0].cpu().numpy())   # CLS position

    H = [np.concatenate(b) for b in buf]              # list of (n_probe, d_model)
    split = int(0.7 * n_probe)

    results = {}
    for name, y in targets.items():
        r2s = []
        for l in range(n_depths):
            H_tr, H_te = H[l][:split], H[l][split:]
            y_tr, y_te = y[:split],    y[split:]
            reg   = Ridge(alpha=1.0).fit(H_tr, y_tr)
            y_hat = reg.predict(H_te)
            if y.ndim == 1:
                r2 = r2_score(y_te, y_hat)
            else:
                # Average R² over frequency bins (skip DC k=0)
                r2 = float(np.mean([max(r2_score(y_te[:, k], y_hat[:, k]), -0.1)
                                    for k in range(1, y.shape[1])]))
            r2s.append(r2)
        results[name] = r2s
    return results


# ─────────────────────────────────────────────────────────────
# 5b. PER-FREQUENCY PROBE  (h[k] → target[k], averaged over k)
# ─────────────────────────────────────────────────────────────

def run_perfreq_probes(model, val_ds, n_probe=3000, n_freq_sample=64):
    """
    For each depth d and sampled frequency bin k:
      - feature: h[d, k]  ∈ ℝ^{d_model}  (hidden state at freq-token position k)
      - target:  y[k]     ∈ ℝ              (cross_re[k], phat_cos[k], …)

    Fit a separate Ridge for each k, then average R² over k.
    This is the KEY probe: cross_re is BILINEAR in the raw input features, so
    R²(Embed → cross_re) ≈ 0, but a jump at L1 would indicate GCC computation.
    """
    model.eval().to(device)
    rng = np.random.RandomState(7)
    idx = rng.choice(len(val_ds), n_probe, replace=False)
    targets = build_probe_targets(val_ds, idx)

    toks_all = val_ds.tokens[idx].to(device)
    n_depths = model.n_layers + 1
    buf = [[] for _ in range(n_depths)]
    BS  = 256
    with torch.no_grad():
        for s in range(0, n_probe, BS):
            _, hs = model(toks_all[s:s+BS], return_hidden=True, include_embed=True)
            for l, h in enumerate(hs):
                buf[l].append(h[:, 1:].cpu().numpy())  # freq tokens (skip CLS)
                                                        # shape (B, F, d_model)

    H = [np.concatenate(b) for b in buf]   # list of (n_probe, F, d_model)

    # Sample frequency bins k = 1..F-1  (skip DC)
    freq_bins = rng.choice(range(1, val_ds.F),
                           min(n_freq_sample, val_ds.F - 1), replace=False)
    split = int(0.7 * n_probe)

    results = {}
    for name in ['cross_re', 'cross_im', 'phat_cos', 'phat_sin']:
        y = targets[name]   # (n_probe, F)
        r2s = []
        for l in range(n_depths):
            r2_per_k = []
            for k in freq_bins:
                H_tr = H[l][:split, k, :]    # (split, d_model)
                H_te = H[l][split:, k, :]
                y_tr = y[:split, k]           # (split,)
                y_te = y[split:, k]
                reg  = Ridge(alpha=1.0).fit(H_tr, y_tr)
                r2_per_k.append(max(r2_score(y_te, reg.predict(H_te)), -0.1))
            r2s.append(float(np.mean(r2_per_k)))
        results[name] = r2s
    return results


# ─────────────────────────────────────────────────────────────
# 6. PLOTTING
# ─────────────────────────────────────────────────────────────

STYLE = {
    'cross_re': ('darkorange',  's-',  'GCC cross-power Re'),
    'cross_im': ('sienna',      'D--', 'GCC cross-power Im'),
    'phat_cos': ('royalblue',   'o-',  'PHAT phase cos'),
    'phat_sin': ('deepskyblue', 'v--', 'PHAT phase sin'),
    'tau':      ('black',       '^-',  'TDOA τ'),
}


def plot_perfreq_analysis(cls_probes, freq_probes, n_layers, exp_label, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    x_ticks   = list(range(n_layers + 1))
    x_labels  = ['Embed'] + [f'L{i+1}' for i in range(n_layers)]

    # ── Panel A: Per-frequency token probe ──
    ax = axes[0]
    for name in ['cross_re', 'cross_im', 'phat_cos', 'phat_sin']:
        c, sty, lbl = STYLE[name]
        ax.plot(x_ticks, freq_probes[name], sty, color=c, lw=2, ms=8, label=lbl)
    ax.axhline(0, c='gray', lw=0.8, ls=':')
    ax.set_title('Per-Frequency Token Probe\nh[k] → target[k]  (averaged over k)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Depth'); ax.set_ylabel('R²')
    ax.set_xticks(x_ticks); ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # annotation: expected GCC step
    for l_idx, l_lbl in [(1, 'GCC\ncross-power'), (2, 'PHAT\nphase')]:
        ax.axvline(l_idx - 0.5, c='green', lw=1, ls='--', alpha=0.5)
    ax.text(0.5, 0.92, 'expected GCC here →', transform=ax.transAxes,
            ha='right', va='top', color='green', fontsize=8)

    # ── Panel B: CLS token probe (multi-target) ──
    ax = axes[1]
    for name in ['cross_re', 'cross_im', 'phat_cos', 'phat_sin']:
        c, sty, lbl = STYLE[name]
        ax.plot(x_ticks, cls_probes[name], sty, color=c, lw=2, ms=8, label=lbl)
    ax.axhline(0, c='gray', lw=0.8, ls=':')
    ax.set_title('CLS Token Probe\nh[CLS] → averaged targets\n(global aggregation only)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Depth'); ax.set_ylabel('R²')
    ax.set_xticks(x_ticks); ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Panel C: TDOA decoding + comparison ──
    ax = axes[2]
    c, sty, lbl = STYLE['tau']
    ax.plot(x_ticks, cls_probes['tau'], sty, color=c, lw=2.5, ms=9, label='CLS → τ (TDOA)')
    ax.plot(x_ticks, freq_probes['phat_cos'], 'o-',  color='royalblue', lw=2, ms=7,
            label='per-freq → PHAT cos')
    ax.plot(x_ticks, cls_probes['phat_cos'],  'o--', color='royalblue', lw=1.5, ms=7,
            alpha=0.5, label='CLS → PHAT cos')
    ax.plot(x_ticks, freq_probes['cross_re'], 's-',  color='darkorange', lw=2, ms=7,
            label='per-freq → cross_re')
    ax.plot(x_ticks, cls_probes['cross_re'],  's--', color='darkorange', lw=1.5, ms=7,
            alpha=0.5, label='CLS → cross_re')
    ax.axhline(0, c='gray', lw=0.8, ls=':')
    ax.set_title('Per-Freq vs CLS Comparison\n(solid = per-freq, dashed = CLS)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Depth'); ax.set_ylabel('R²')
    ax.set_xticks(x_ticks); ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(
        f'GCC-PHAT Mechanistic Interpretability — Per-Frequency Probe Analysis\n'
        f'{exp_label}  |  Token = [Re(X1[k]), Im(X1[k]), Re(X2[k]), Im(X2[k])]',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {path}")
    plt.close()


def print_summary(tag, cls_p, freq_p, n_layers):
    depth_lbl = ['Embed'] + [f'L{i+1}' for i in range(n_layers)]
    print(f"\n  {'─'*52}")
    print(f"  {tag}")
    print(f"  {'─'*52}")
    print(f"  {'Depth':8s}  {'per-freq cross_re':>18s}  {'CLS cross_re':>13s}  "
          f"{'per-freq phat_cos':>18s}  {'CLS tau':>8s}")
    for i, lbl in enumerate(depth_lbl):
        print(f"  {lbl:8s}  "
              f"{freq_p['cross_re'][i]:>18.3f}  "
              f"{cls_p['cross_re'][i]:>13.3f}  "
              f"{freq_p['phat_cos'][i]:>18.3f}  "
              f"{cls_p['tau'][i]:>8.3f}")


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    T, TAU_MAX   = 256, 30
    N_TRAIN, N_VAL = 50_000, 5_000
    EPOCHS       = 120

    EXPERIMENTS = [
        dict(noise_type='colored', snr_db=0.0),   # clearest result from initial run
        dict(noise_type='white',   snr_db=0.0),   # comparison
    ]

    for cfg in EXPERIMENTS:
        tag = f"{cfg['noise_type']}_snr{cfg['snr_db']:+.0f}dB"
        exp_label = f"noise={cfg['noise_type']}, SNR={cfg['snr_db']:+.0f} dB"
        print(f"\n{'━'*60}")
        print(f"  Experiment: {tag}")
        print(f"{'━'*60}")

        print("  [1/5] Data ...")
        t0    = time.time()
        tr_ds = TDOADataset(N_TRAIN, TAU_MAX, cfg['snr_db'], T, cfg['noise_type'], seed=42)
        va_ds = TDOADataset(N_VAL,   TAU_MAX, cfg['snr_db'], T, cfg['noise_type'], seed=99)
        print(f"       {time.time()-t0:.1f}s")

        print("  [2/5] Model ...")
        model = TDOATransformer(F=tr_ds.F, d_model=64, n_layers=4, n_heads=4, d_ff=256)
        print(f"       params: {model.n_params:,}")

        print("  [3/5] Train ...")
        t0   = time.time()
        hist = train_model(model, tr_ds, va_ds, epochs=EPOCHS, lr=3e-4, batch_size=1024)
        val_mae = hist['val_mae'][-1]
        print(f"       {time.time()-t0:.1f}s | final val MAE: {val_mae:.3f} samples")
        torch.save(model.state_dict(), f'results/{tag}_model.pt')

        print("  [4/5] CLS probes ...")
        cls_p = run_cls_probes(model, va_ds, n_probe=3000)
        for nm, r2s in cls_p.items():
            print(f"       CLS  {nm:12s}: {[f'{v:.3f}' for v in r2s]}")

        print("  [5/5] Per-frequency probes ...")
        freq_p = run_perfreq_probes(model, va_ds, n_probe=3000, n_freq_sample=64)
        for nm, r2s in freq_p.items():
            print(f"       Freq {nm:12s}: {[f'{v:.3f}' for v in r2s]}")

        print_summary(tag, cls_p, freq_p, model.n_layers)
        plot_perfreq_analysis(cls_p, freq_p, model.n_layers, exp_label,
                              path=f'results/perfreq_probe_{tag}.png')

        torch.save({'cfg': cfg, 'history': hist,
                    'val_mae': val_mae,
                    'cls_probes': cls_p, 'freq_probes': freq_p},
                   f'results/perfreq_{tag}.pt')

    print(f"\n{'━'*60}")
    print("ALL DONE")
    print(f"{'━'*60}")
    print("Figures: results/perfreq_probe_*.png")
    print("Data:    results/perfreq_*.pt")
