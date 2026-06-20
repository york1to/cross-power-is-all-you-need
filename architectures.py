#!/usr/bin/env python3
"""
Cross-Architecture Mechanistic Comparison for TDOA Estimation
=============================================================
Core question: Is the cross-power spectrum representation universal?

GCC-PHAT step 1:  cross_re[k] = Re(X1[k]·X2[k]*) — bilinear in input
If this intermediate representation appears after the first nonlinear layer
in ALL architectures, it is a "necessary" computation forced by the physics
of the problem, not a Transformer-specific artifact.

Three architectures with matched parameter budgets (~200k params):
  MLP-bin   : Per-frequency MLP (independent FFN per bin) + global linear readout
              No explicit cross-freq interaction until the final aggregation layer
  CNN-freq  : 1D-CNN along the frequency axis (local mixing) + global avg pool
              Local frequency neighbourhood interaction
  Transformer: Our existing model (global attention across all frequencies)

Probing methodology: identical to perfreq_probe.py
  - Per-frequency probe: h[depth, k] → target[k]  (averaged over k)
  - CLS/global probe:    h[depth, global] → tau

Prediction:
  - cross_re/im per-freq R² jumps to >0.8 at depth 1 for ALL architectures
    → cross-power is physics-forced, not architecture-specific
  - phat_cos stays low for all → PHAT whitening is not explicitly encoded
  - tau probe: Transformer builds it through layers; MLP-bin needs final aggregation
    layer; CNN-freq through progressive mixing → different aggregation pathways
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")
os.makedirs('results', exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. DATA  (reuse from perfreq_probe.py)
# ─────────────────────────────────────────────────────────────

from perfreq_probe import (colored_noise, make_pair, to_tokens,
                            TDOADataset, build_probe_targets)


# ─────────────────────────────────────────────────────────────
# 2. ARCHITECTURES
# ─────────────────────────────────────────────────────────────

# ── 2a. Per-frequency MLP ─────────────────────────────────────
class MLPBinModel(nn.Module):
    """
    For each frequency bin k independently: MLP(4→d→d→d) on the token [X1_re, X1_im, X2_re, X2_im].
    Then a linear aggregation over all F bin representations → TDOA.

    Probing note: "per-frequency hidden states" are the outputs of each MLP layer.
    The global aggregation (final linear) is probed via the mean-pooled representation.

    Depth checkpoints: [embed, mlp_l1, mlp_l2, mlp_l3, aggregate]
    """
    def __init__(self, F, d=64, n_layers=3):
        super().__init__()
        self.F, self.d, self.n_layers = F, d, n_layers
        # Shared weights across frequency bins (weight tying = regularisation)
        layers = [nn.Linear(4, d), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward_hidden(self, x):
        """x: (B, Fq, 4) → list of (B, Fq, d) pre-activation at each linear layer"""
        B, Fq, _ = x.shape
        h = x.reshape(B * Fq, 4)
        hiddens = []
        w = list(self.mlp.children())          # alternating Linear, GELU, ...
        # First linear (pre-activation = embedding checkpoint)
        h_lin = w[0](h)
        hiddens.append(h_lin.reshape(B, Fq, -1).detach())
        h_cur = F_nn.gelu(h_lin)
        # Remaining linear+activation pairs
        i = 2
        while i < len(w):
            h_lin = w[i](h_cur)                # linear
            hiddens.append(h_lin.reshape(B, Fq, -1).detach())
            h_cur = F_nn.gelu(h_lin)           # activation
            i += 2
        return hiddens  # n_layers tensors of shape (B, Fq, d)

    def forward(self, x, return_hidden=False):
        B, Fq, _ = x.shape
        h = x.reshape(B * Fq, 4)
        h = self.mlp(h)                        # (B*Fq, d)
        h = h.reshape(B, Fq, self.d)           # (B, Fq, d)
        h_pool = self.norm(h.mean(dim=1))      # (B, d)
        out = self.head(h_pool)
        if return_hidden:
            hh = self.forward_hidden(x)
            return out, hh, h_pool
        return out

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)



# ── 2b. 1D-CNN along frequency axis ──────────────────────────
class CNNFreqModel(nn.Module):
    """
    1D-CNN along the frequency dimension.
    Input: (B, 4, F) — treat 4 features as channels, F as sequence length.
    Architecture: 4 Conv1d layers (same padding), kernel=5 → gradually mix
    adjacent frequency bins. Final global average pool → TDOA.

    Depth checkpoints: embed (first conv output) + subsequent conv outputs.
    """
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
                nn.GELU()          # F here is the constructor arg (freq bins), not F_nn
            ))
            in_ch = d
        self.head = nn.Linear(d, 1)

    def forward(self, x, return_hidden=False):
        # x: (B, F, 4) → (B, 4, F)
        h = x.permute(0, 2, 1)
        hiddens = []
        for layer in self.layers_:
            h = layer(h)
            if return_hidden:
                hiddens.append(h.permute(0, 2, 1).detach())  # (B, F, d)
        out = self.head(h.mean(dim=-1))                       # global avg pool over F
        return (out, hiddens) if return_hidden else out

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 2c. Transformer (identical to perfreq_probe.py) ──────────
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


# ─────────────────────────────────────────────────────────────
# 3. TRAINING  (generic)
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
# 4. PROBING  (architecture-agnostic)
# ─────────────────────────────────────────────────────────────

def collect_hidden_states(model, toks_all, n_probe, arch):
    """
    Returns:
      freq_H: list of (n_probe, F, d)  — one per depth checkpoint
      global_H: list of (n_probe, d)   — global/CLS token per depth
    """
    BS = 256
    if arch == 'transformer':
        n_depths = model.n_layers + 1   # embed + 4 layers
        f_buf  = [[] for _ in range(n_depths)]
        g_buf  = [[] for _ in range(n_depths)]
        with torch.no_grad():
            for s in range(0, n_probe, BS):
                _, hs = model(toks_all[s:s+BS], return_hidden=True, include_embed=True)
                for l, h in enumerate(hs):
                    f_buf[l].append(h[:, 1:].cpu().numpy())   # freq tokens
                    g_buf[l].append(h[:, 0].cpu().numpy())    # CLS
        freq_H   = [np.concatenate(b) for b in f_buf]
        global_H = [np.concatenate(b) for b in g_buf]

    elif arch == 'cnn':
        n_depths = model.n_layers
        f_buf  = [[] for _ in range(n_depths)]
        with torch.no_grad():
            for s in range(0, n_probe, BS):
                _, hs = model(toks_all[s:s+BS], return_hidden=True)
                for l, h in enumerate(hs):
                    f_buf[l].append(h.cpu().numpy())           # (B, F, d)
        freq_H   = [np.concatenate(b) for b in f_buf]
        global_H = [h.mean(axis=1) for h in freq_H]           # global avg pool

    elif arch == 'mlp_bin':
        # Use the forward_hidden method
        n_depths = model.n_layers   # embed_linear + (n_layers-1) post-linear
        f_buf = [[] for _ in range(n_depths)]
        with torch.no_grad():
            for s in range(0, n_probe, BS):
                hs = model.forward_hidden(toks_all[s:s+BS])
                for l, h in enumerate(hs):
                    f_buf[l].append(h.cpu().numpy())
        freq_H   = [np.concatenate(b) for b in f_buf]
        global_H = [h.mean(axis=1) for h in freq_H]           # mean over freq bins

    return freq_H, global_H


def run_arch_probes(model, val_ds, arch, n_probe=3000, n_freq_sample=64):
    model.eval().to(device)
    rng = np.random.RandomState(7)
    idx = rng.choice(len(val_ds), n_probe, replace=False)
    targets = build_probe_targets(val_ds, idx)
    freq_bins = rng.choice(range(1, val_ds.F),
                           min(n_freq_sample, val_ds.F - 1), replace=False)
    split = int(0.7 * n_probe)

    toks_all = val_ds.tokens[idx].to(device)
    freq_H, global_H = collect_hidden_states(model, toks_all, n_probe, arch)
    n_depths = len(freq_H)

    # Per-frequency probes
    freq_results = {}
    for name in ['cross_re', 'cross_im', 'phat_cos', 'phat_sin']:
        y  = targets[name]   # (n_probe, F)
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

    # Global/CLS probes
    global_results = {}
    for name, y in targets.items():
        r2s = []
        for l in range(n_depths):
            H_tr, H_te = global_H[l][:split], global_H[l][split:]
            y_tr, y_te = y[:split],           y[split:]
            reg   = Ridge(alpha=1.0).fit(H_tr, y_tr)
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
# 5. PLOTTING
# ─────────────────────────────────────────────────────────────

ARCH_COLORS = {
    'mlp_bin':     ('royalblue',   'o-',  'MLP-per-bin'),
    'cnn':         ('darkorange',  's-',  '1D-CNN (freq)'),
    'transformer': ('crimson',     '^-',  'Transformer'),
}


def plot_arch_comparison(arch_results, exp_label, path):
    """
    3 rows × 4 cols (one col per architecture):
    Row 0: per-freq cross_re R² vs depth
    Row 1: per-freq phat_cos R² vs depth
    Row 2: global tau R² vs depth (+ cross_re comparison)
    """
    arch_names = list(arch_results.keys())
    n_archs = len(arch_names)

    fig, axes = plt.subplots(3, n_archs, figsize=(6*n_archs, 14))
    if n_archs == 1: axes = axes[:, None]

    for col, arch in enumerate(arch_names):
        res = arch_results[arch]
        n_depths = res['n_depths']
        x = list(range(n_depths))
        x_lbls = [f'D{i}' for i in range(n_depths)]
        if arch == 'transformer':
            x_lbls = ['Emb'] + [f'L{i+1}' for i in range(n_depths-1)]
        elif arch == 'mlp_bin':
            x_lbls = ['Lin'] + [f'L{i+1}' for i in range(n_depths-1)]
        c, sty, lbl = ARCH_COLORS[arch]

        # Row 0: per-freq cross_re + cross_im
        ax = axes[0, col]
        ax.plot(x, res['freq']['cross_re'], 'o-', color='darkorange', lw=2, ms=8,
                label='cross_re')
        ax.plot(x, res['freq']['cross_im'], 's--', color='sienna', lw=1.5, ms=7,
                label='cross_im')
        ax.axhline(0, c='gray', lw=0.8, ls=':')
        ax.set_title(f'{lbl}\nPer-freq: GCC cross-power R²', fontsize=10, fontweight='bold')
        ax.set_xlabel('Depth'); ax.set_ylabel('R²')
        ax.set_xticks(x); ax.set_xticklabels(x_lbls)
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.text(0.5, 0.05, f'val MAE={res["val_mae"]:.2f}',
                transform=ax.transAxes, ha='center', fontsize=9, color='gray')

        # Row 1: per-freq phat_cos + phat_sin
        ax = axes[1, col]
        ax.plot(x, res['freq']['phat_cos'], 'o-', color='royalblue', lw=2, ms=8,
                label='phat_cos')
        ax.plot(x, res['freq']['phat_sin'], 'v--', color='deepskyblue', lw=1.5, ms=7,
                label='phat_sin')
        ax.axhline(0, c='gray', lw=0.8, ls=':')
        ax.set_title(f'{lbl}\nPer-freq: PHAT phase R²', fontsize=10, fontweight='bold')
        ax.set_xlabel('Depth'); ax.set_ylabel('R²')
        ax.set_xticks(x); ax.set_xticklabels(x_lbls)
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Row 2: global tau + cross_re comparison
        ax = axes[2, col]
        ax.plot(x, res['global']['tau'], '^-', color='black', lw=2.5, ms=9,
                label='Global → τ')
        ax.plot(x, res['freq']['cross_re'], 'o--', color='darkorange', lw=1.5, ms=7,
                alpha=0.7, label='per-freq cross_re')
        ax.axhline(0, c='gray', lw=0.8, ls=':')
        ax.set_title(f'{lbl}\nGlobal τ decode + cross_re', fontsize=10, fontweight='bold')
        ax.set_xlabel('Depth'); ax.set_ylabel('R²')
        ax.set_xticks(x); ax.set_xticklabels(x_lbls)
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(
        f'Cross-Architecture Mechanistic Comparison\n{exp_label}\n'
        f'All: same data, same probing pipeline — "Is cross-power universal?"',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {path}")
    plt.close()


def plot_summary(arch_results, exp_label, path):
    """Overlay all 3 architectures on 2 panels: cross_re and tau R² vs depth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for arch, res in arch_results.items():
        c, sty, lbl = ARCH_COLORS[arch]
        n = res['n_depths']
        x = list(range(n))
        axes[0].plot(x, res['freq']['cross_re'], sty, color=c, lw=2.5, ms=9,
                     label=f'{lbl} (val MAE={res["val_mae"]:.2f})')
        axes[1].plot(x, res['global']['tau'],    sty, color=c, lw=2.5, ms=9,
                     label=f'{lbl} (val MAE={res["val_mae"]:.2f})')

    for ax, title in zip(axes,
                         ['Per-Frequency Cross-Power R²\n(Bilinear → Linear after first nonlinearity)',
                          'Global TDOA Decode R²\n(Aggregation pathway)']):
        ax.axhline(0, c='gray', lw=0.8, ls=':')
        ax.set_xlabel('Depth'); ax.set_ylabel('R²')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(-0.15, 1.05)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(
        f'Cross-Architecture Summary — {exp_label}\n'
        f'Key claim: cross-power R² jump is universal (physics-forced), '
        f'aggregation varies by architecture',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────

T, TAU_MAX   = 256, 30
N_TRAIN, N_VAL = 50_000, 5_000
EPOCHS       = 120
F            = T//2+1                 # 129

# Run on colored noise 0dB (highest performance gap, clearest signal)
CFG = dict(noise_type='colored', snr_db=0.0)
tag = f"{CFG['noise_type']}_snr{CFG['snr_db']:+.0f}dB"
exp_label = f"noise={CFG['noise_type']}, SNR={CFG['snr_db']:+.0f} dB"

print(f"\n{'━'*60}")
print(f"  Cross-architecture comparison: {tag}")
print(f"{'━'*60}")

print("  [1/4] Data ...")
tr_ds = TDOADataset(N_TRAIN, TAU_MAX, CFG['snr_db'], T, CFG['noise_type'], seed=42)
va_ds = TDOADataset(N_VAL,   TAU_MAX, CFG['snr_db'], T, CFG['noise_type'], seed=99)

ARCHITECTURES = [
    ('mlp_bin',     MLPBinModel(F=F, d=64, n_layers=3)),
    ('cnn',         CNNFreqModel(F=F, d=64, n_layers=4, kernel=5)),
    ('transformer', TransformerModel(F=F, d_model=64, n_layers=4, n_heads=4, d_ff=256)),
]

arch_results = {}

for arch_name, model in ARCHITECTURES:
    print(f"\n  ── Architecture: {arch_name} ──")
    print(f"     params: {model.n_params:,}")

    print("  [2/4] Train ...")
    t0   = time.time()
    hist = train_model(model, tr_ds, va_ds, epochs=EPOCHS, lr=3e-4, batch_size=1024)
    val_mae = hist['val_mae'][-1]
    print(f"     {time.time()-t0:.1f}s | final val MAE: {val_mae:.3f}")
    torch.save(model.state_dict(), f'results/arch_{arch_name}_{tag}_model.pt')

    print("  [3/4] Probing ...")
    freq_p, global_p, n_depths = run_arch_probes(model, va_ds, arch_name,
                                                  n_probe=3000, n_freq_sample=64)
    for name in ['cross_re', 'phat_cos']:
        print(f"     per-freq {name:12s}: {[f'{v:.3f}' for v in freq_p[name]]}")
    print(f"     global   tau         : {[f'{v:.3f}' for v in global_p['tau']]}")

    arch_results[arch_name] = {
        'freq': freq_p, 'global': global_p,
        'n_depths': n_depths, 'val_mae': val_mae, 'history': hist
    }
    torch.save(arch_results[arch_name],
               f'results/arch_{arch_name}_{tag}.pt')

print("\n  [4/4] Plotting ...")
plot_arch_comparison(arch_results, exp_label,
                     f'results/arch_comparison_{tag}.png')
plot_summary(arch_results, exp_label,
             f'results/arch_summary_{tag}.png')

# Summary table
print(f"\n{'━'*60}")
print("ARCHITECTURE COMPARISON SUMMARY")
print(f"{'━'*60}")
arch_param_map = {name: model.n_params for name, model in ARCHITECTURES}
print(f"{'Arch':15s}  {'params':>8s}  {'val MAE':>8s}  {'cross_re@D1':>12s}  {'tau@last':>9s}")
for arch_name, res in arch_results.items():
    print(f"{arch_name:15s}  "
          f"{arch_param_map[arch_name]:>8,}  "
          f"{res['val_mae']:>8.3f}  "
          f"{res['freq']['cross_re'][1]:>12.3f}  "
          f"{res['global']['tau'][-1]:>9.3f}")

print(f"\nFigures: results/arch_*.png")
print(f"Data:    results/arch_*.pt")
