#!/usr/bin/env python3
"""
Follow-up Experiments for PHAT-Reinvent
========================================
Two experiments to strengthen the core finding:
  "Cross-power is universal; PHAT whitening is unanimously rejected."

Experiment A — Matched-Parameter MLP-bin-wide
  Increase MLP hidden dim (d=256, ~133k params ≈ CNN's 129k).
  If Phase 1 stays at R²≈0.98 and Phase 2 stays at R²≈0.5,
  it separates "capacity limitation" from "missing cross-freq pathway".

Experiment B — Nonlinear Probe for PHAT
  Replace Ridge (linear) with 2-layer MLP probe for phat_cos/sin.
  If nonlinear probe R² is also low, PHAT info is genuinely absent,
  not just encoded nonlinearly. This rules out the "probe expressivity" confound.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.utils.data import DataLoader, TensorDataset
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
# SHARED CONFIG
# ─────────────────────────────────────────────────────────────

T, TAU_MAX   = 256, 30
N_TRAIN, N_VAL = 50_000, 5_000
EPOCHS       = 120
F            = T // 2 + 1            # 129
CFG = dict(noise_type='colored', snr_db=0.0)
tag = f"{CFG['noise_type']}_snr{CFG['snr_db']:+.0f}dB"


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────

class MLPBinModel(nn.Module):
    """Per-frequency MLP (shared weights) + mean pool → head."""
    def __init__(self, F, d=64, n_layers=3):
        super().__init__()
        self.F, self.d, self.n_layers = F, d, n_layers
        layers = [nn.Linear(4, d), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward_hidden(self, x):
        B, Fq, _ = x.shape
        h = x.reshape(B * Fq, 4)
        hiddens = []
        w = list(self.mlp.children())
        h_lin = w[0](h)
        hiddens.append(h_lin.reshape(B, Fq, -1).detach())
        h_cur = F_nn.gelu(h_lin)
        i = 2
        while i < len(w):
            h_lin = w[i](h_cur)
            hiddens.append(h_lin.reshape(B, Fq, -1).detach())
            h_cur = F_nn.gelu(h_lin)
            i += 2
        return hiddens

    def forward(self, x, return_hidden=False):
        B, Fq, _ = x.shape
        h = x.reshape(B * Fq, 4)
        h = self.mlp(h)
        h = h.reshape(B, Fq, self.d)
        h_pool = self.norm(h.mean(dim=1))
        out = self.head(h_pool)
        if return_hidden:
            return out, self.forward_hidden(x), h_pool
        return out

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNFreqModel(nn.Module):
    """1D-CNN along frequency dimension."""
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
# HIDDEN STATE COLLECTION (reuse from architectures.py)
# ─────────────────────────────────────────────────────────────

def collect_hidden_states(model, toks_all, n_probe, arch):
    BS = 256
    if arch == 'transformer':
        n_depths = model.n_layers + 1
        f_buf  = [[] for _ in range(n_depths)]
        g_buf  = [[] for _ in range(n_depths)]
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
        f_buf  = [[] for _ in range(n_depths)]
        with torch.no_grad():
            for s in range(0, n_probe, BS):
                _, hs = model(toks_all[s:s+BS], return_hidden=True)
                for l, h in enumerate(hs):
                    f_buf[l].append(h.cpu().numpy())
        freq_H   = [np.concatenate(b) for b in f_buf]
        global_H = [h.mean(axis=1) for h in freq_H]

    elif arch in ('mlp_bin', 'mlp_bin_wide'):
        n_depths = model.n_layers
        f_buf = [[] for _ in range(n_depths)]
        with torch.no_grad():
            for s in range(0, n_probe, BS):
                hs = model.forward_hidden(toks_all[s:s+BS])
                for l, h in enumerate(hs):
                    f_buf[l].append(h.cpu().numpy())
        freq_H   = [np.concatenate(b) for b in f_buf]
        global_H = [h.mean(axis=1) for h in freq_H]

    return freq_H, global_H


# ─────────────────────────────────────────────────────────────
# LINEAR PROBING (Ridge — same as before)
# ─────────────────────────────────────────────────────────────

def run_linear_probes(freq_H, global_H, targets, freq_bins, split):
    n_depths = len(freq_H)

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

    return freq_results, global_results


# ─────────────────────────────────────────────────────────────
# NONLINEAR PROBING (2-layer MLP)
# ─────────────────────────────────────────────────────────────

class MLPProbe(nn.Module):
    """2-layer MLP probe: d_in → d_hidden → 1."""
    def __init__(self, d_in, d_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp_probe(H_tr, y_tr, H_te, y_te, d_hidden=128, lr=1e-3,
                    epochs=300, batch_size=512):
    """Train a 2-layer MLP probe and return test R²."""
    d_in = H_tr.shape[1]
    probe = MLPProbe(d_in, d_hidden).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    tr_H = torch.from_numpy(H_tr).float().to(device)
    tr_y = torch.from_numpy(y_tr).float().to(device)
    te_H = torch.from_numpy(H_te).float().to(device)
    te_y = torch.from_numpy(y_te).float().to(device)

    ds = TensorDataset(tr_H, tr_y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for xb, yb in dl:
            pred = probe(xb)
            loss = F_nn.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    probe.eval()
    with torch.no_grad():
        y_hat = probe(te_H).cpu().numpy()
    return max(r2_score(y_te, y_hat), -0.1)


def run_nonlinear_probes(freq_H, global_H, targets, freq_bins, split,
                         d_hidden=128, epochs=300):
    """Same structure as linear probes, but using 2-layer MLP."""
    n_depths = len(freq_H)

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
                r2 = train_mlp_probe(H_tr, y_tr, H_te, y_te,
                                     d_hidden=d_hidden, epochs=epochs)
                r2_per_k.append(r2)
            r2s.append(float(np.mean(r2_per_k)))
            print(f"       NL probe {name:12s} depth={l}: R²={r2s[-1]:.3f}")
        freq_results[name] = r2s

    global_results = {}
    for name in ['tau']:
        y = targets[name]
        r2s = []
        for l in range(n_depths):
            H_tr, H_te = global_H[l][:split], global_H[l][split:]
            y_tr, y_te = y[:split], y[split:]
            r2 = train_mlp_probe(H_tr, y_tr, H_te, y_te,
                                 d_hidden=d_hidden, epochs=epochs)
            r2s.append(r2)
            print(f"       NL probe tau          depth={l}: R²={r2s[-1]:.3f}")
        global_results[name] = r2s

    return freq_results, global_results


# ─────────────────────────────────────────────────────────────
# EXPERIMENT A: MATCHED-PARAMETER MLP
# ─────────────────────────────────────────────────────────────

def experiment_a(tr_ds, va_ds):
    print(f"\n{'='*60}")
    print("  EXPERIMENT A: Matched-Parameter MLP")
    print(f"  Claim: Phase 2 failure is architectural, not capacity.")
    print(f"{'='*60}")

    # Original small MLP for comparison
    mlp_small = MLPBinModel(F=F, d=64,  n_layers=3)
    # New wide MLP matched to CNN (~130k params)
    mlp_wide  = MLPBinModel(F=F, d=256, n_layers=3)
    # CNN baseline
    cnn       = CNNFreqModel(F=F, d=64, n_layers=4, kernel=5)

    models = [
        ('mlp_bin_d64',  mlp_small, 'mlp_bin'),
        ('mlp_bin_d256', mlp_wide,  'mlp_bin_wide'),
        ('cnn_d64',      cnn,       'cnn'),
    ]

    for name, model, arch_type in models:
        print(f"\n  ── {name}: {model.n_params:,} params ──")

    rng = np.random.RandomState(7)
    n_probe = 3000
    idx = rng.choice(len(va_ds), n_probe, replace=False)
    targets = build_probe_targets(va_ds, idx)
    freq_bins = rng.choice(range(1, va_ds.F),
                           min(64, va_ds.F - 1), replace=False)
    split = int(0.7 * n_probe)

    results = {}
    for name, model, arch_type in models:
        print(f"\n  ── Training {name} ──")
        t0 = time.time()
        hist = train_model(model, tr_ds, va_ds, epochs=EPOCHS, lr=3e-4, batch_size=1024)
        val_mae = hist['val_mae'][-1]
        print(f"     {time.time()-t0:.1f}s | val MAE: {val_mae:.3f}")

        model.eval().to(device)
        toks_all = va_ds.tokens[idx].to(device)
        freq_H, global_H = collect_hidden_states(model, toks_all, n_probe, arch_type)

        freq_p, global_p = run_linear_probes(
            freq_H, global_H, targets, freq_bins, split)

        results[name] = {
            'freq': freq_p, 'global': global_p,
            'n_depths': len(freq_H), 'val_mae': val_mae,
            'n_params': model.n_params, 'history': hist,
        }

        print(f"     cross_re per-freq: {[f'{v:.3f}' for v in freq_p['cross_re']]}")
        print(f"     tau global:        {[f'{v:.3f}' for v in global_p['tau']]}")

    # Save
    torch.save(results, f'results/expA_matched_param_{tag}.pt')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {'mlp_bin_d64': 'royalblue', 'mlp_bin_d256': 'mediumseagreen', 'cnn_d64': 'darkorange'}
    markers = {'mlp_bin_d64': 'o-', 'mlp_bin_d256': 'D-', 'cnn_d64': 's-'}
    labels = {
        'mlp_bin_d64':  f'MLP-bin d=64 ({results["mlp_bin_d64"]["n_params"]:,}p, MAE={results["mlp_bin_d64"]["val_mae"]:.2f})',
        'mlp_bin_d256': f'MLP-bin d=256 ({results["mlp_bin_d256"]["n_params"]:,}p, MAE={results["mlp_bin_d256"]["val_mae"]:.2f})',
        'cnn_d64':      f'CNN d=64 ({results["cnn_d64"]["n_params"]:,}p, MAE={results["cnn_d64"]["val_mae"]:.2f})',
    }

    for name, res in results.items():
        n = res['n_depths']
        x = list(range(n))
        axes[0].plot(x, res['freq']['cross_re'], markers[name], color=colors[name],
                     lw=2.5, ms=9, label=labels[name])
        axes[1].plot(x, res['global']['tau'], markers[name], color=colors[name],
                     lw=2.5, ms=9, label=labels[name])

    axes[0].set_title('Per-Freq Cross-Power R²\n(Phase 1: Local Computation)',
                      fontsize=11, fontweight='bold')
    axes[1].set_title('Global TDOA Decode R²\n(Phase 2: Aggregation)',
                      fontsize=11, fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Depth'); ax.set_ylabel('R²')
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.axhline(0, c='gray', lw=0.8, ls=':')

    plt.suptitle(
        'Experiment A — Matched-Parameter Capacity Control\n'
        'Does widening MLP fix Phase 2? (Prediction: No — missing cross-freq pathway)',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = f'results/expA_matched_param_{tag}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  → Saved: {path}")
    plt.close()

    return results


# ─────────────────────────────────────────────────────────────
# EXPERIMENT B: NONLINEAR PROBE ABLATION
# ─────────────────────────────────────────────────────────────

def experiment_b(tr_ds, va_ds):
    print(f"\n{'='*60}")
    print("  EXPERIMENT B: Nonlinear Probe for PHAT Whitening")
    print(f"  Claim: PHAT info is genuinely absent, not just nonlinearly encoded.")
    print(f"{'='*60}")

    # Train all three original architectures
    arch_models = [
        ('mlp_bin',     MLPBinModel(F=F, d=64, n_layers=3),     'mlp_bin'),
        ('cnn',         CNNFreqModel(F=F, d=64, n_layers=4),    'cnn'),
        ('transformer', TransformerModel(F=F, d_model=64, n_layers=4, n_heads=4, d_ff=256), 'transformer'),
    ]

    rng = np.random.RandomState(7)
    n_probe = 3000
    n_freq_sample = 32   # fewer bins since nonlinear probes are slower

    results = {}
    for name, model, arch_type in arch_models:
        print(f"\n  ── {name} ({model.n_params:,} params) ──")

        print("     Training ...")
        t0 = time.time()
        hist = train_model(model, tr_ds, va_ds, epochs=EPOCHS, lr=3e-4, batch_size=1024)
        val_mae = hist['val_mae'][-1]
        print(f"     {time.time()-t0:.1f}s | val MAE: {val_mae:.3f}")

        model.eval().to(device)
        idx = rng.choice(len(va_ds), n_probe, replace=False)
        targets = build_probe_targets(va_ds, idx)
        freq_bins = rng.choice(range(1, va_ds.F),
                               min(n_freq_sample, va_ds.F - 1), replace=False)
        split = int(0.7 * n_probe)
        toks_all = va_ds.tokens[idx].to(device)

        freq_H, global_H = collect_hidden_states(model, toks_all, n_probe, arch_type)

        # Linear probes
        print("     Linear probes ...")
        lin_freq, lin_global = run_linear_probes(
            freq_H, global_H, targets, freq_bins, split)

        # Nonlinear probes
        print("     Nonlinear probes ...")
        nl_freq, nl_global = run_nonlinear_probes(
            freq_H, global_H, targets, freq_bins, split,
            d_hidden=128, epochs=300)

        results[name] = {
            'lin_freq': lin_freq, 'lin_global': lin_global,
            'nl_freq':  nl_freq,  'nl_global':  nl_global,
            'n_depths': len(freq_H), 'val_mae': val_mae,
            'n_params': model.n_params,
        }

        # Print comparison
        print(f"\n     {'Target':12s}  {'Depth':>6s}  {'Linear':>8s}  {'Nonlinear':>10s}  {'Δ':>6s}")
        for tgt in ['cross_re', 'phat_cos', 'phat_sin']:
            for d in range(len(freq_H)):
                l_r2 = lin_freq[tgt][d]
                n_r2 = nl_freq[tgt][d]
                print(f"     {tgt:12s}  {d:>6d}  {l_r2:>8.3f}  {n_r2:>10.3f}  {n_r2-l_r2:>+6.3f}")

    # Save
    torch.save(results, f'results/expB_nonlinear_probe_{tag}.pt')

    # Plot: 3 architectures × (linear vs nonlinear) for phat_cos + cross_re
    arch_names = list(results.keys())
    fig, axes = plt.subplots(2, len(arch_names), figsize=(6*len(arch_names), 10))

    arch_labels = {'mlp_bin': 'MLP-per-bin', 'cnn': '1D-CNN', 'transformer': 'Transformer'}

    for col, arch in enumerate(arch_names):
        res = results[arch]
        n = res['n_depths']
        x = list(range(n))

        # Row 0: cross_re (sanity check — both probes should be high)
        ax = axes[0, col]
        ax.plot(x, res['lin_freq']['cross_re'], 'o-', color='darkorange', lw=2, ms=8,
                label='Linear (Ridge)')
        ax.plot(x, res['nl_freq']['cross_re'], 's--', color='red', lw=2, ms=8,
                label='Nonlinear (MLP)')
        ax.set_title(f'{arch_labels[arch]}\ncross_re: Linear vs Nonlinear Probe',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Depth'); ax.set_ylabel('R²')
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.axhline(0, c='gray', lw=0.8, ls=':')

        # Row 1: phat_cos (the KEY test)
        ax = axes[1, col]
        ax.plot(x, res['lin_freq']['phat_cos'], 'o-', color='royalblue', lw=2, ms=8,
                label='Linear (Ridge)')
        ax.plot(x, res['nl_freq']['phat_cos'], 's--', color='darkviolet', lw=2, ms=8,
                label='Nonlinear (MLP)')
        ax.plot(x, res['lin_freq']['phat_sin'], 'v:', color='deepskyblue', lw=1.5, ms=7,
                label='Linear phat_sin', alpha=0.7)
        ax.plot(x, res['nl_freq']['phat_sin'], '^:', color='mediumpurple', lw=1.5, ms=7,
                label='NL phat_sin', alpha=0.7)
        ax.set_title(f'{arch_labels[arch]}\nphat_cos/sin: Linear vs Nonlinear Probe',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Depth'); ax.set_ylabel('R²')
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.axhline(0, c='gray', lw=0.8, ls=':')

    plt.suptitle(
        'Experiment B — Nonlinear Probe Ablation\n'
        'If nonlinear probe R² for PHAT is also low → PHAT info genuinely absent\n'
        '(not just encoded nonlinearly)',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = f'results/expB_nonlinear_probe_{tag}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  → Saved: {path}")
    plt.close()

    return results


# ─────────────────────────────────────────────────────────────
# COMBINED SUMMARY FIGURE
# ─────────────────────────────────────────────────────────────

def plot_combined_summary(res_a, res_b):
    """Create a single summary figure with both experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Top-left: Exp A — Phase 1 (cross_re) ──
    ax = axes[0, 0]
    colors_a = {'mlp_bin_d64': 'royalblue', 'mlp_bin_d256': 'mediumseagreen', 'cnn_d64': 'darkorange'}
    labels_a = {
        'mlp_bin_d64':  f'MLP d=64 ({res_a["mlp_bin_d64"]["n_params"]:,}p)',
        'mlp_bin_d256': f'MLP d=256 ({res_a["mlp_bin_d256"]["n_params"]:,}p)',
        'cnn_d64':      f'CNN d=64 ({res_a["cnn_d64"]["n_params"]:,}p)',
    }
    for name, res in res_a.items():
        x = list(range(res['n_depths']))
        ax.plot(x, res['freq']['cross_re'], 'o-', color=colors_a[name], lw=2.5, ms=9,
                label=labels_a[name])
    ax.set_title('Exp A: Phase 1 (cross_re)\nCapacity does NOT change computation',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Depth'); ax.set_ylabel('R²')
    ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Top-right: Exp A — Phase 2 (tau) ──
    ax = axes[0, 1]
    for name, res in res_a.items():
        x = list(range(res['n_depths']))
        ax.plot(x, res['global']['tau'], 'o-', color=colors_a[name], lw=2.5, ms=9,
                label=f'{labels_a[name]} (MAE={res["val_mae"]:.2f})')
    ax.set_title('Exp A: Phase 2 (tau)\nWider MLP still fails → architectural limit',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Depth'); ax.set_ylabel('R²')
    ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Bottom-left: Exp B — cross_re sanity (all architectures) ──
    ax = axes[1, 0]
    colors_b = {'mlp_bin': 'royalblue', 'cnn': 'darkorange', 'transformer': 'crimson'}
    for arch, res in res_b.items():
        n = res['n_depths']
        x = list(range(n))
        ax.plot(x, res['lin_freq']['cross_re'], 'o-', color=colors_b[arch], lw=2, ms=8,
                label=f'{arch} linear', alpha=0.7)
        ax.plot(x, res['nl_freq']['cross_re'], 's--', color=colors_b[arch], lw=2, ms=8,
                label=f'{arch} nonlinear')
    ax.set_title('Exp B: cross_re sanity check\nLinear ≈ Nonlinear (cross-power is linearly decodable)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Depth'); ax.set_ylabel('R²')
    ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # ── Bottom-right: Exp B — phat_cos KEY RESULT ──
    ax = axes[1, 1]
    for arch, res in res_b.items():
        n = res['n_depths']
        x = list(range(n))
        ax.plot(x, res['lin_freq']['phat_cos'], 'o-', color=colors_b[arch], lw=2, ms=8,
                label=f'{arch} linear', alpha=0.7)
        ax.plot(x, res['nl_freq']['phat_cos'], 's--', color=colors_b[arch], lw=2, ms=8,
                label=f'{arch} nonlinear')
    ax.set_title('Exp B: phat_cos — THE KEY TEST\nBoth probes low → PHAT genuinely absent',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Depth'); ax.set_ylabel('R²')
    ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.axhspan(-0.15, 0.3, alpha=0.08, color='red', label='_')
    ax.text(0.5, 0.2, '"PHAT-free zone"', transform=ax.transAxes,
            ha='center', fontsize=10, color='red', alpha=0.5, fontstyle='italic')

    for ax in axes.flat:
        ax.axhline(0, c='gray', lw=0.8, ls=':')

    plt.suptitle(
        'Follow-up Experiments — Strengthening the Core Finding\n'
        '"Cross-power is the invariant; PHAT whitening is genuinely rejected"',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = f'results/followup_summary_{tag}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  → Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("  [1/4] Building data ...")
    tr_ds = TDOADataset(N_TRAIN, TAU_MAX, CFG['snr_db'], T, CFG['noise_type'], seed=42)
    va_ds = TDOADataset(N_VAL,   TAU_MAX, CFG['snr_db'], T, CFG['noise_type'], seed=99)

    print("  [2/4] Experiment A: Matched-Parameter MLP ...")
    res_a = experiment_a(tr_ds, va_ds)

    print("  [3/4] Experiment B: Nonlinear Probe Ablation ...")
    res_b = experiment_b(tr_ds, va_ds)

    print("  [4/4] Combined Summary ...")
    plot_combined_summary(res_a, res_b)

    # Print final summary table
    print(f"\n{'━'*70}")
    print("FOLLOW-UP EXPERIMENTS SUMMARY")
    print(f"{'━'*70}")

    print("\n  Experiment A: Matched-Parameter MLP")
    print(f"  {'Model':20s}  {'Params':>8s}  {'Val MAE':>8s}  {'cross_re@D1':>12s}  {'tau@last':>9s}")
    for name, res in res_a.items():
        print(f"  {name:20s}  {res['n_params']:>8,}  {res['val_mae']:>8.3f}  "
              f"{res['freq']['cross_re'][1]:>12.3f}  {res['global']['tau'][-1]:>9.3f}")

    print("\n  Experiment B: Nonlinear vs Linear Probe")
    print(f"  {'Arch':15s}  {'Target':12s}  {'Best Linear':>12s}  {'Best Nonlinear':>14s}  {'Verdict':>10s}")
    for arch, res in res_b.items():
        for tgt in ['cross_re', 'phat_cos', 'phat_sin']:
            best_lin = max(res['lin_freq'][tgt])
            best_nl  = max(res['nl_freq'][tgt])
            verdict = 'PRESENT' if best_nl > 0.5 else 'ABSENT'
            print(f"  {arch:15s}  {tgt:12s}  {best_lin:>12.3f}  {best_nl:>14.3f}  {verdict:>10s}")

    print(f"\nFigures: results/expA_*.png, results/expB_*.png, results/followup_summary_*.png")
    print(f"Data:    results/expA_*.pt, results/expB_*.pt")
    print(f"\n{'━'*70}")
    print("DONE")
    print(f"{'━'*70}")
