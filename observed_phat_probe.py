#!/usr/bin/env python3
"""
Experiment: Observed PHAT Probing
=================================
Addresses reviewer comment: "Probing using observed PHAT (G/|G|) would
strengthen the claim that PHAT is absent."

Current probes use *theoretical* PHAT targets: cos(2πkτ/N), sin(2πkτ/N).
These are the noiseless ideal. If the network computes observed (noisy)
PHAT, i.e. G12/|G12|, the theoretical probe would miss it.

This experiment adds two new probe targets:
  obs_phat_re[k] = Re(G12[k] / |G12[k]|)
  obs_phat_im[k] = Im(G12[k] / |G12[k]|)

If the network computes observed PHAT, these should be decodable even
when theoretical PHAT is not.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
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

    def forward(self, x, return_hidden=False):
        h = x.permute(0, 2, 1)
        hiddens = []
        for layer in self.layers_:
            h = layer(h)
            if return_hidden:
                hiddens.append(h.permute(0, 2, 1).detach())
        out = self.head(h.mean(dim=-1))
        return (out, hiddens) if return_hidden else out


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
N_PROBE = 3000
N_FREQ_SAMPLE = 64


def build_extended_targets(val_ds, idx):
    """Build all probe targets including observed PHAT."""
    T, F = val_ds.T, val_ds.F
    toks = val_ds.tokens[idx].numpy()
    taus = val_ds.taus_raw[idx].numpy()

    X1 = toks[:, :, 0] + 1j * toks[:, :, 1]
    X2 = toks[:, :, 2] + 1j * toks[:, :, 3]
    cross = X1 * np.conj(X2)

    # Theoretical PHAT (noiseless ideal)
    k = np.arange(F)[None, :]
    phi = 2 * np.pi * k * taus[:, None] / T

    # Observed PHAT: G12 / |G12|
    cross_mag = np.abs(cross) + 1e-10
    obs_phat = cross / cross_mag

    return {
        'tau':         (taus / val_ds.tau_max).astype(np.float32),
        'cross_re':    cross.real.astype(np.float32),
        'cross_im':    cross.imag.astype(np.float32),
        'phat_cos':    np.cos(phi).astype(np.float32),
        'phat_sin':    np.sin(phi).astype(np.float32),
        'obs_phat_re': obs_phat.real.astype(np.float32),
        'obs_phat_im': obs_phat.imag.astype(np.float32),
        'cross_mag':   cross_mag.astype(np.float32),
    }


def collect_hidden_states(model, toks_all, n_probe, arch):
    """Collect per-frequency hidden states at all depths."""
    BS = 256
    model.eval()

    if arch == 'transformer':
        n_depths = model.n_layers + 1
        buf = [[] for _ in range(n_depths)]
        with torch.no_grad():
            for s in range(0, n_probe, BS):
                _, hs = model(toks_all[s:s+BS], return_hidden=True, include_embed=True)
                for l, h in enumerate(hs):
                    buf[l].append(h[:, 1:].cpu().numpy())
    elif arch == 'cnn':
        n_depths = model.n_layers
        buf = [[] for _ in range(n_depths)]
        with torch.no_grad():
            for s in range(0, n_probe, BS):
                _, hs = model(toks_all[s:s+BS], return_hidden=True)
                for l, h in enumerate(hs):
                    buf[l].append(h.cpu().numpy())
    elif arch == 'mlp_bin':
        n_depths = model.n_layers
        buf = [[] for _ in range(n_depths)]
        with torch.no_grad():
            for s in range(0, n_probe, BS):
                hs = model.forward_hidden(toks_all[s:s+BS])
                for l, h in enumerate(hs):
                    buf[l].append(h.cpu().numpy())
    else:
        raise ValueError(f"Unknown arch: {arch}")

    return [np.concatenate(b) for b in buf]


def run_probes(H_list, targets, val_ds, target_names):
    """Run per-frequency Ridge probes for given targets."""
    rng = np.random.RandomState(7)
    freq_bins = rng.choice(range(1, val_ds.F),
                           min(N_FREQ_SAMPLE, val_ds.F - 1), replace=False)
    split = int(0.7 * len(H_list[0]))

    results = {}
    for name in target_names:
        y = targets[name]
        r2s = []
        for l in range(len(H_list)):
            r2_per_k = []
            for k in freq_bins:
                H_tr = H_list[l][:split, k, :]
                H_te = H_list[l][split:, k, :]
                y_tr = y[:split, k]
                y_te = y[split:, k]
                reg = Ridge(alpha=1.0).fit(H_tr, y_tr)
                r2_per_k.append(max(r2_score(y_te, reg.predict(H_te)), -0.1))
            r2s.append(float(np.mean(r2_per_k)))
        results[name] = r2s
    return results


def plot_comparison(results, arch, condition, path):
    """Plot theoretical vs observed PHAT probing comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_depths = len(results['cross_re'])
    if arch == 'transformer':
        x_labels = ['Emb'] + [f'L{i+1}' for i in range(n_depths - 1)]
    else:
        x_labels = [f'L{i+1}' for i in range(n_depths)]
    x = list(range(n_depths))

    # Panel A: All targets comparison
    ax = axes[0]
    ax.plot(x, results['cross_re'], 's-', color='darkorange', lw=2, ms=8,
            label='cross_re (cross-power)')
    ax.plot(x, results['phat_cos'], 'o-', color='royalblue', lw=2, ms=8,
            label='phat_cos (theoretical)')
    ax.plot(x, results['obs_phat_re'], 'D-', color='crimson', lw=2, ms=8,
            label='obs_phat_re (observed)')
    ax.plot(x, results['cross_mag'], '^-', color='gray', lw=2, ms=8,
            label='cross_mag (|G12|)')
    ax.axhline(0, c='gray', lw=0.8, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_ylabel('R²'); ax.set_xlabel('Depth')
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title(f'Probing comparison ({arch}, {condition})', fontweight='bold')

    # Panel B: Theoretical vs Observed PHAT zoom
    ax = axes[1]
    ax.plot(x, results['phat_cos'], 'o-', color='royalblue', lw=2, ms=8,
            label='Theoretical PHAT cos')
    ax.plot(x, results['phat_sin'], 'v--', color='deepskyblue', lw=2, ms=7,
            label='Theoretical PHAT sin')
    ax.plot(x, results['obs_phat_re'], 'D-', color='crimson', lw=2, ms=8,
            label='Observed PHAT Re')
    ax.plot(x, results['obs_phat_im'], 'D--', color='salmon', lw=2, ms=7,
            label='Observed PHAT Im')
    ax.axhline(0, c='gray', lw=0.8, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_ylabel('R²'); ax.set_xlabel('Depth')
    ax.set_ylim(-0.15, 0.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title('Theoretical vs Observed PHAT (zoomed)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ── Main ──
if __name__ == '__main__':
    CONDITIONS = [
        dict(noise_type='colored', snr_db=0.0),
        dict(noise_type='white',   snr_db=0.0),
        dict(noise_type='colored', snr_db=-5.0),
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

            # Build extended targets
            rng = np.random.RandomState(7)
            idx = rng.choice(len(va_ds), N_PROBE, replace=False)
            targets = build_extended_targets(va_ds, idx)
            toks_all = va_ds.tokens[idx].to(device)

            # Collect hidden states
            H_list = collect_hidden_states(model, toks_all, N_PROBE, arch_name)

            # Run probes on all targets
            target_names = ['cross_re', 'cross_im', 'phat_cos', 'phat_sin',
                            'obs_phat_re', 'obs_phat_im', 'cross_mag']
            results = run_probes(H_list, targets, va_ds, target_names)

            # Print
            n_depths = len(results['cross_re'])
            if arch_name == 'transformer':
                depth_labels = ['Emb'] + [f'L{i+1}' for i in range(n_depths - 1)]
            else:
                depth_labels = [f'L{i+1}' for i in range(n_depths)]

            print(f"\n  {'Depth':6s}  {'cross_re':>9s}  {'phat_cos':>9s}  "
                  f"{'obs_phat_re':>12s}  {'cross_mag':>10s}")
            for i, lbl in enumerate(depth_labels):
                print(f"  {lbl:6s}  {results['cross_re'][i]:>9.3f}  "
                      f"{results['phat_cos'][i]:>9.3f}  "
                      f"{results['obs_phat_re'][i]:>12.3f}  "
                      f"{results['cross_mag'][i]:>10.3f}")

            key = f"{tag}_{arch_name}"
            all_results[key] = results
            plot_comparison(results, arch_name, condition,
                            f'results/obs_phat_{key}.png')

    # Save all results
    torch.save(all_results, 'results/observed_phat_probing.pt')
    print(f"\n{'='*60}")
    print("ALL DONE — results/observed_phat_probing.pt")
    print(f"{'='*60}")
