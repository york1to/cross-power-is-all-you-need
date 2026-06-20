#!/usr/bin/env python3
"""
Generate 5 scatter plots (y_true vs y_pred) for each probe target,
showing the geometric intuition behind R².

Uses the saved Transformer model (colored, 0dB) and re-runs Ridge probing
to capture individual predictions.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Data (copied to avoid importing perfreq_probe.py) ──

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
        self.tokens = torch.from_numpy(toks)
        self.taus_norm = torch.from_numpy(taus / tau_max)
        self.taus_raw = torch.from_numpy(taus)
        self.T, self.F, self.tau_max = T, T//2+1, tau_max
    def __len__(self): return len(self.taus_norm)
    def __getitem__(self, i): return self.tokens[i], self.taus_norm[i:i+1]

# ── Model ──

class TDOATransformer(nn.Module):
    def __init__(self, F, d_model=64, n_layers=4, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.F, self.d_model, self.n_layers = F, d_model, n_layers
        self.input_proj = nn.Linear(4, d_model)
        self.freq_embed = nn.Embedding(F, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers_ = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, return_hidden=False, include_embed=False):
        B = x.shape[0]
        h = self.input_proj(x)
        h = h + self.freq_embed(torch.arange(self.F, device=x.device))
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        hiddens = []
        if return_hidden and include_embed:
            hiddens.append(h.detach())
        for layer in self.layers_:
            h = layer(h)
            if return_hidden:
                hiddens.append(h.detach())
        out = self.head(self.norm_out(h[:, 0]))
        return (out, hiddens) if return_hidden else out

# ── Probe targets ──

def build_probe_targets(val_ds, idx):
    T, F = val_ds.T, val_ds.F
    toks = val_ds.tokens[idx].numpy()
    taus = val_ds.taus_raw[idx].numpy()
    X1 = toks[:, :, 0] + 1j*toks[:, :, 1]
    X2 = toks[:, :, 2] + 1j*toks[:, :, 3]
    cross = X1 * np.conj(X2)
    k = np.arange(F)[None, :]
    phi = 2*np.pi * k * taus[:, None] / T
    return {
        'tau':      (taus / val_ds.tau_max).astype(np.float32),
        'cross_re': cross.real.astype(np.float32),
        'cross_im': cross.imag.astype(np.float32),
        'phat_cos': np.cos(phi).astype(np.float32),
        'phat_sin': np.sin(phi).astype(np.float32),
    }

# ── Main ──

if __name__ == '__main__':
    T, TAU_MAX = 256, 30
    N_VAL = 5_000
    n_probe = 3000
    BEST_LAYER = 4  # L4 (index 4 = Embed + 4 layers)
    FREQ_BIN = 20   # pick a mid-frequency bin for per-freq probes

    print("Loading validation data...")
    va_ds = TDOADataset(N_VAL, TAU_MAX, snr_db=0.0, T=T,
                        noise_type='colored', seed=99)

    print("Loading model...")
    model = TDOATransformer(F=va_ds.F, d_model=64, n_layers=4, n_heads=4, d_ff=256)
    model.load_state_dict(torch.load('results/colored_snr+0dB_model.pt',
                                      map_location=device))
    model.eval().to(device)

    # Sample probe indices
    rng = np.random.RandomState(7)
    idx = rng.choice(len(va_ds), n_probe, replace=False)
    targets = build_probe_targets(va_ds, idx)

    # Extract hidden states at best layer
    print("Extracting hidden states...")
    toks_all = va_ds.tokens[idx].to(device)
    cls_buf, freq_buf = [], []
    BS = 256
    with torch.no_grad():
        for s in range(0, n_probe, BS):
            _, hs = model(toks_all[s:s+BS], return_hidden=True, include_embed=True)
            h = hs[BEST_LAYER]  # L4
            cls_buf.append(h[:, 0].cpu().numpy())           # CLS token
            freq_buf.append(h[:, 1+FREQ_BIN].cpu().numpy()) # freq token at bin k

    H_cls = np.concatenate(cls_buf)    # (n_probe, 64)
    H_freq = np.concatenate(freq_buf)  # (n_probe, 64)
    split = int(0.7 * n_probe)

    # ── Generate 5 scatter plots ──
    plot_specs = [
        ('cross_re', H_freq, targets['cross_re'][:, FREQ_BIN],
         r'$\mathrm{Re}(X_1 \cdot X_2^*)$', f'Per-freq probe at k={FREQ_BIN}'),
        ('cross_im', H_freq, targets['cross_im'][:, FREQ_BIN],
         r'$\mathrm{Im}(X_1 \cdot X_2^*)$', f'Per-freq probe at k={FREQ_BIN}'),
        ('phat_cos', H_freq, targets['phat_cos'][:, FREQ_BIN],
         r'$\cos(2\pi k \tau / T)$', f'Per-freq probe at k={FREQ_BIN}'),
        ('phat_sin', H_freq, targets['phat_sin'][:, FREQ_BIN],
         r'$\sin(2\pi k \tau / T)$', f'Per-freq probe at k={FREQ_BIN}'),
        ('tau',      H_cls,  targets['tau'],
         r'$\tau$ (TDOA)', 'CLS probe'),
    ]

    print("Fitting probes and plotting...")
    for name, H, y, ylabel, subtitle in plot_specs:
        H_tr, H_te = H[:split], H[split:]
        y_tr, y_te = y[:split], y[split:]
        reg = Ridge(alpha=1.0).fit(H_tr, y_tr)
        y_hat = reg.predict(H_te)
        r2 = r2_score(y_te, y_hat)

        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        # Scatter
        ax.scatter(y_te, y_hat, s=6, alpha=0.3, color='steelblue',
                   edgecolors='none', rasterized=True)

        # Perfect fit line
        lo, hi = min(y_te.min(), y_hat.min()), max(y_te.max(), y_hat.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'r--', lw=1.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        out_path = f'results/scatter_{name}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print(f"  {name:10s}  R²={r2:.3f}  → {out_path}")

    print("\nDone.")
