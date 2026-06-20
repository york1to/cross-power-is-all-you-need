#!/usr/bin/env python3
"""
GCC-PHAT Mechanistic Interpretability — Initial Validation
============================================================
Input design (key choice):
  For each frequency bin k, one token = [Re(X1[k]), Im(X1[k]), Re(X2[k]), Im(X2[k])]
  → F=129 tokens, sequence length F+1 (with CLS)
  → FFN in layer 0 can immediately compute cross-power; attention aggregates across freqs
  → This matches the algorithmic structure of GCC-PHAT

GCC-PHAT steps we probe for:
  Step 1 — cross_re/im:  Re/Im(X1·X2*)   raw cross-power  (before whitening)
  Step 2 — phat_cos/sin: cos/sin(2πkτ/T) PHAT-normalised phase
  Step 3 — tau:          τ (final output)
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
import matplotlib.gridspec as gridspec
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────

def colored_noise(T, beta=1.0, rng=None):
    rng = rng or np.random
    f = np.fft.rfftfreq(T); f[0] = 1.0
    pwr = f ** (-beta / 2); pwr[0] = 0.0
    return np.fft.irfft((rng.randn(len(f)) + 1j*rng.randn(len(f))) * pwr, T).astype(np.float32)

def make_pair(tau, snr_db, T=256, noise_type='white', rng=None):
    rng = rng or np.random
    src = (rng.randn(T) if noise_type == 'white' else colored_noise(T, 1.0, rng)).astype(np.float32)
    src /= np.std(src) + 1e-8
    S = np.fft.rfft(src)
    k = np.arange(len(S))
    mic2_clean = np.fft.irfft(S * np.exp(-1j * 2*np.pi*k*tau/T), T).astype(np.float32)
    if np.isinf(snr_db):
        return src.copy(), mic2_clean
    ns = 10**(-snr_db/20)
    return src + rng.randn(T).astype(np.float32)*ns, mic2_clean + rng.randn(T).astype(np.float32)*ns

def to_tokens(mic1, mic2):
    """
    Returns (F, 4) — one token per frequency bin.
    token[k] = [Re(X1[k]), Im(X1[k]), Re(X2[k]), Im(X2[k])]
    Normalised so std ≈ 1 (phases preserved).
    """
    X1, X2 = np.fft.rfft(mic1), np.fft.rfft(mic2)
    F = len(X1)
    t = np.stack([X1.real, X1.imag, X2.real, X2.imag], axis=1).astype(np.float32)  # (F,4)
    return t / (np.std(t) + 1e-8)

class TDOADataset(Dataset):
    def __init__(self, N, tau_max=30, snr_db=0., T=256, noise_type='white', seed=0):
        rng = np.random.RandomState(seed)
        taus = rng.uniform(-tau_max, tau_max, N).astype(np.float32)
        toks = np.stack([to_tokens(*make_pair(t, snr_db, T, noise_type, rng))
                         for t in tqdm(taus, desc='  data', leave=False)])
        self.tokens   = torch.from_numpy(toks)          # (N, F, 4)
        self.taus_norm= torch.from_numpy(taus/tau_max)  # (N,)  ∈ [-1,1]
        self.taus_raw = torch.from_numpy(taus)          # (N,)  samples
        self.T, self.F, self.tau_max = T, T//2+1, tau_max

    def __len__(self): return len(self.taus_norm)
    def __getitem__(self, i):
        return self.tokens[i], self.taus_norm[i:i+1]   # (F,4), (1,)


# ─────────────────────────────────────────────────────────────
# 2. MODEL
# ─────────────────────────────────────────────────────────────

class TDOATransformer(nn.Module):
    def __init__(self, F, d_model=64, n_layers=4, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.F, self.d_model, self.n_layers = F, d_model, n_layers
        self.input_proj  = nn.Linear(4, d_model)
        self.freq_embed  = nn.Embedding(F, d_model)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers_     = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, 1)

    def _embed(self, x):
        B = x.shape[0]
        h = self.input_proj(x)                                     # (B, F, d)
        h = h + self.freq_embed(torch.arange(self.F, device=x.device))  # +pos
        cls = self.cls_token.expand(B, -1, -1)                    # (B,1,d)
        return torch.cat([cls, h], dim=1)                          # (B, F+1, d)

    def forward(self, x, return_hidden=False):
        h = self._embed(x)
        hiddens = []
        for layer in self.layers_:
            h = layer(h)
            if return_hidden:
                hiddens.append(h.detach())
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
    warmup_steps = int(warmup_frac * epochs * len(tr))
    total_steps  = epochs * len(tr)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sch  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.HuberLoss()
    tau_max = train_ds.tau_max
    history = {'train_mae': [], 'val_mae': []}

    pbar = tqdm(range(epochs), desc='  train')
    step = 0
    for epoch in pbar:
        model.train()
        tr_mae = []
        for toks, tau in tr:
            toks, tau = toks.to(device), tau.to(device)
            pred = model(toks)
            loss = loss_fn(pred, tau)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); step += 1
            tr_mae.append((pred-tau).abs().mean().item() * tau_max)

        model.eval()
        va_mae = []
        with torch.no_grad():
            for toks, tau in va:
                toks, tau = toks.to(device), tau.to(device)
                va_mae.append((model(toks)-tau).abs().mean().item() * tau_max)

        tm, vm = np.mean(tr_mae), np.mean(va_mae)
        history['train_mae'].append(tm); history['val_mae'].append(vm)
        pbar.set_postfix({'tr': f'{tm:.3f}', 'va': f'{vm:.3f}',
                          'lr': f'{sch.get_last_lr()[0]:.2e}'})
    return history


# ─────────────────────────────────────────────────────────────
# 4. GCC-PHAT BASELINE
# ─────────────────────────────────────────────────────────────

def gcc_phat_eval(val_ds, n=500):
    T, F = val_ds.T, val_ds.F
    maes = []
    for i in range(min(n, len(val_ds))):
        toks = val_ds.tokens[i].numpy()          # (F, 4)
        tau_true = val_ds.taus_raw[i].item()
        X1 = toks[:, 0] + 1j*toks[:, 1]
        X2 = toks[:, 2] + 1j*toks[:, 3]
        gcc = X1 * np.conj(X2)
        phat = np.fft.irfft(gcc / (np.abs(gcc)+1e-10), T)
        est = int(np.argmax(np.abs(phat)))
        if est > T//2: est -= T
        maes.append(abs(est - tau_true))
    return float(np.mean(maes))


# ─────────────────────────────────────────────────────────────
# 5. LINEAR PROBING
# ─────────────────────────────────────────────────────────────

def build_probe_targets(val_ds, idx):
    """
    Returns dict of regression targets:
      tau          : scalar per sample
      cross_re/im  : Re/Im of X1·X2* per frequency (F values)  — GCC step
      phat_cos/sin : cos/sin(2πkτ/T)  per frequency            — PHAT step
    """
    T, F = val_ds.T, val_ds.F
    toks = val_ds.tokens[idx].numpy()          # (N, F, 4)
    taus = val_ds.taus_raw[idx].numpy()        # (N,)

    X1   = toks[:, :, 0] + 1j*toks[:, :, 1]  # (N, F)
    X2   = toks[:, :, 2] + 1j*toks[:, :, 3]  # (N, F)
    cross = X1 * np.conj(X2)                   # (N, F)

    k   = np.arange(F)[None, :]               # (1, F)
    phi = 2*np.pi * k * taus[:, None] / T     # (N, F)

    return {
        'tau':       (taus / val_ds.tau_max).astype(np.float32),
        'cross_re':  cross.real.astype(np.float32),
        'cross_im':  cross.imag.astype(np.float32),
        'phat_cos':  np.cos(phi).astype(np.float32),
        'phat_sin':  np.sin(phi).astype(np.float32),
    }

def run_probes(model, val_ds, n_probe=3000):
    """
    For each layer, train a linear Ridge probe from CLS hidden state → each target.
    Returns {probe_name: [R²_layer1, R²_layer2, ...]}
    """
    model.eval().to(device)
    idx = np.random.RandomState(7).choice(len(val_ds), n_probe, replace=False)
    targets = build_probe_targets(val_ds, idx)

    # Collect per-layer CLS hidden states
    toks_all = val_ds.tokens[idx].to(device)
    buf = [[] for _ in range(model.n_layers)]
    BS  = 256
    with torch.no_grad():
        for s in range(0, n_probe, BS):
            _, hs = model(toks_all[s:s+BS], return_hidden=True)
            for l, h in enumerate(hs):
                buf[l].append(h[:, 0].cpu().numpy())   # CLS token

    H = [np.concatenate(b) for b in buf]               # list[(n_probe, d)]
    split = int(0.7 * n_probe)

    results = {}
    for name, y in targets.items():
        r2s = []
        for l in range(model.n_layers):
            H_tr, H_te = H[l][:split], H[l][split:]
            y_tr, y_te = y[:split], y[split:]
            reg = Ridge(alpha=1.0).fit(H_tr, y_tr)
            y_hat = reg.predict(H_te)
            if y.ndim == 1:
                r2 = r2_score(y_te, y_hat)
            else:
                r2 = float(np.mean([max(r2_score(y_te[:,k], y_hat[:,k]), -0.1)
                                    for k in range(1, y.shape[1])]))
            r2s.append(r2)
        results[name] = r2s
    return results


# ─────────────────────────────────────────────────────────────
# 6. ATTENTION PATTERNS
# ─────────────────────────────────────────────────────────────

def get_attn(model, toks, layer_idx=0):
    model.eval()
    with torch.no_grad():
        h = model._embed(toks.to(device))
        for i, layer in enumerate(model.layers_):
            if i == layer_idx:
                hn = layer.norm1(h)
                _, w = layer.self_attn(hn, hn, hn, need_weights=True,
                                       average_attn_weights=False)
                return w.cpu().numpy()   # (B, heads, seq, seq)
            h = layer(h)
    return None


# ─────────────────────────────────────────────────────────────
# 7. PLOTTING
# ─────────────────────────────────────────────────────────────

PROBE_STYLE = {
    'tau':       ('TDOA (direct)',        'black',      'o-',  2.0),
    'cross_re':  ('GCC cross-power Re',   'darkorange', 's--', 1.5),
    'phat_cos':  ('PHAT phase cos [key]', 'royalblue',  '^-',  2.0),
    'phat_sin':  ('PHAT phase sin',       'deepskyblue','v-',  1.5),
}

def plot_all(results, path='results/validation_results.png'):
    n = len(results)
    fig = plt.figure(figsize=(7*n, 13))
    gs  = gridspec.GridSpec(3, n, hspace=0.5, wspace=0.35)

    for col, res in enumerate(results):
        lbl = f"{res['noise_type']} | SNR={res['snr_db']:+.0f}dB"

        # Row 0: training curve
        ax = fig.add_subplot(gs[0, col])
        ax.semilogy(res['history']['train_mae'], label='Train')
        ax.semilogy(res['history']['val_mae'],   label='Val')
        ax.axhline(res['gcc_mae'], c='red', ls='--',
                   label=f'GCC-PHAT ({res["gcc_mae"]:.2f})')
        ax.set_title(f'Training Curve\n{lbl}', fontsize=10)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAE (samples)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

        # Row 1: probe R²
        ax = fig.add_subplot(gs[1, col])
        layers = list(range(1, res['n_layers']+1))
        for pname, (plbl, col_, sty, lw) in PROBE_STYLE.items():
            if pname in res['probes']:
                ax.plot(layers, res['probes'][pname], sty,
                        label=plbl, color=col_, lw=lw, markersize=7)
        ax.axhline(0, c='gray', lw=0.8, ls=':')
        ax.set_title(f'Linear Probe R²\n{lbl}', fontsize=10)
        ax.set_xlabel('Layer'); ax.set_ylabel('R²')
        ax.set_xticks(layers); ax.set_ylim(-0.15, 1.05)
        ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

        # Row 2: attention heatmap (layer 0, head 0)
        ax = fig.add_subplot(gs[2, col])
        attn = res.get('attn_l0')
        if attn is not None:
            a = attn[0, 0, 1:, 1:]                         # drop CLS row/col
            step = max(1, a.shape[0]//64)
            im = ax.imshow(a[::step, ::step], cmap='hot', aspect='auto', vmin=0)
            plt.colorbar(im, ax=ax, fraction=0.04)
            ax.set_title(f'Attention L0 H0\n{lbl}', fontsize=9)
            ax.set_xlabel('Key (freq bin)'); ax.set_ylabel('Query (freq bin)')

    plt.suptitle(
        'GCC-PHAT Mechanistic Interpretability — Initial Validation\n'
        'Token = [Re(X1[k]), Im(X1[k]), Re(X2[k]), Im(X2[k])]  •  F=129 freq tokens',
        fontsize=12, fontweight='bold')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────

EXPERIMENTS = [
    dict(noise_type='white',   snr_db=0.0),
    dict(noise_type='colored', snr_db=0.0),
    dict(noise_type='colored', snr_db=-5.0),
]
T, TAU_MAX = 256, 30
N_TRAIN, N_VAL = 50_000, 5_000
EPOCHS = 120

all_results = []

for cfg in EXPERIMENTS:
    tag = f"{cfg['noise_type']}_snr{cfg['snr_db']:+.0f}dB"
    print(f"\n{'━'*60}")
    print(f"  Experiment: {tag}")
    print(f"{'━'*60}")

    print("  [1/5] Data ...")
    t0 = time.time()
    tr_ds = TDOADataset(N_TRAIN, TAU_MAX, cfg['snr_db'], T, cfg['noise_type'], seed=42)
    va_ds = TDOADataset(N_VAL,   TAU_MAX, cfg['snr_db'], T, cfg['noise_type'], seed=99)
    print(f"       {time.time()-t0:.1f}s")

    print("  [2/5] Model ...")
    model = TDOATransformer(F=tr_ds.F, d_model=64, n_layers=4, n_heads=4, d_ff=256)
    print(f"       params: {model.n_params:,}")

    print("  [3/5] Train ...")
    t0 = time.time()
    hist = train_model(model, tr_ds, va_ds, epochs=EPOCHS, lr=3e-4, batch_size=1024)
    print(f"       {time.time()-t0:.1f}s | final val MAE: {hist['val_mae'][-1]:.3f} samp")

    print("  [4/5] GCC-PHAT baseline ...")
    gcc = gcc_phat_eval(va_ds, n=500)
    print(f"       GCC-PHAT MAE: {gcc:.3f} samples")

    print("  [5/5] Probing + attention ...")
    probes = run_probes(model, va_ds, n_probe=3000)
    for nm, r2s in probes.items():
        print(f"       {nm:12s}: {[f'{v:.3f}' for v in r2s]}")
    attn_l0 = get_attn(model, va_ds.tokens[:8], layer_idx=0)

    res = {**cfg,
           'history': hist, 'gcc_mae': gcc, 'final_val_mae': hist['val_mae'][-1],
           'probes': probes, 'attn_l0': attn_l0,
           'n_layers': model.n_layers, 'T': T}
    all_results.append(res)
    torch.save(res, f'results/{tag}.pt')

plot_all(all_results)

print(f"\n{'━'*60}")
print("SUMMARY")
print(f"{'━'*60}")
for res in all_results:
    print(f"  {res['noise_type']:7s} SNR={res['snr_db']:+.0f}dB  "
          f"Xfmr: {res['final_val_mae']:.3f}  GCC-PHAT: {res['gcc_mae']:.3f} samples")
print()
print("Probe R² — PHAT phase cos (layer 1→4):")
for res in all_results:
    r2s = res['probes'].get('phat_cos', [])
    print(f"  {res['noise_type']:7s} SNR={res['snr_db']:+.0f}dB: "
          f"{[f'{v:.3f}' for v in r2s]}")
