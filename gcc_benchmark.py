#!/usr/bin/env python3
"""
GCC-Learned vs GCC-PHAT Benchmark
===================================
The key "actionable insight" experiment: if networks reject PHAT and prefer
magnitude-correlated weighting, does using that learned weighting profile
as a classical GCC weight actually beat GCC-PHAT?

Method:
  1. Train Transformer at each condition, extract gradient-based weighting profile
  2. Use the profile as a fixed GCC weighting: R[τ] = IFFT(W_learned[k] · X1·X2*)
  3. Compare TDOA estimation accuracy (MAE in samples) of:
     - GCC-PHAT:      W[k] = 1 / |G12[k]|
     - GCC-Magnitude:  W[k] = |G12[k]|
     - GCC-SCOT:       W[k] = 1 / sqrt(|X1|² · |X2|²)
     - GCC-Flat:       W[k] = 1 (unweighted / basic cross-correlation)
     - GCC-Learned:    W[k] = gradient profile from trained network

  Test across: 6 SNR levels × {white, colored} noise + 3 room conditions
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DATA GENERATION  (self-contained, no perfreq_probe import)
# ─────────────────────────────────────────────────────────────

T, TAU_MAX = 256, 30
F = T // 2 + 1


def colored_noise(T, beta=1.0, rng=None):
    rng = rng or np.random
    f = np.fft.rfftfreq(T); f[0] = 1.0
    pwr = f ** (-beta / 2); pwr[0] = 0.0
    return np.fft.irfft((rng.randn(len(f)) + 1j*rng.randn(len(f))) * pwr, T).astype(np.float32)


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
    ns = 10**(-snr_db/20)
    return (src + rng.randn(T).astype(np.float32)*ns,
            mic2_clean + rng.randn(T).astype(np.float32)*ns)


def to_tokens(mic1, mic2):
    X1, X2 = np.fft.rfft(mic1), np.fft.rfft(mic2)
    t = np.stack([X1.real, X1.imag, X2.real, X2.imag], axis=1).astype(np.float32)
    return t / (np.std(t) + 1e-8)


class TDOADataset:
    def __init__(self, N, tau_max=30, snr_db=0., T=256, noise_type='white', seed=0):
        rng = np.random.RandomState(seed)
        taus = rng.uniform(-tau_max, tau_max, N).astype(np.float32)
        toks = np.stack([to_tokens(*make_pair(t, snr_db, T, noise_type, rng))
                         for t in tqdm(taus, desc='  data', leave=False)])
        self.tokens   = torch.from_numpy(toks)
        self.taus_norm= torch.from_numpy(taus/tau_max)
        self.taus_raw = torch.from_numpy(taus)
        self.T, self.F, self.tau_max = T, T//2+1, tau_max

    def __len__(self): return len(self.taus_norm)
    def __getitem__(self, i):
        return self.tokens[i], self.taus_norm[i:i+1]


# ─────────────────────────────────────────────────────────────
# TRANSFORMER MODEL
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

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_model(model, train_ds, val_ds, epochs=120, lr=3e-4, batch_size=1024):
    model.to(device)
    tr = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    va = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    total_steps  = epochs * len(tr)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * p))

    sch     = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.HuberLoss()
    pbar = tqdm(range(epochs), desc='  train')
    for _ in pbar:
        model.train()
        for toks, tau in tr:
            toks, tau = toks.to(device), tau.to(device)
            pred = model(toks)
            loss = loss_fn(pred, tau)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        model.eval()
        va_mae = []
        with torch.no_grad():
            for toks, tau in va:
                toks, tau = toks.to(device), tau.to(device)
                va_mae.append((model(toks) - tau).abs().mean().item() * val_ds.tau_max)
        pbar.set_postfix({'va': f'{np.mean(va_mae):.3f}'})
    return np.mean(va_mae)


# ─────────────────────────────────────────────────────────────
# GRADIENT-BASED WEIGHTING EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_gradient_profile(model, dataset, n_samples=2000, batch_size=64):
    """Extract average gradient-based frequency weighting profile."""
    model.eval().to(device)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    all_grad_w = []
    for s in range(0, len(idx), batch_size):
        batch_idx = idx[s:s + batch_size]
        toks = dataset.tokens[batch_idx].to(device).requires_grad_(True)
        pred = model(toks)
        pred.sum().backward()
        grad = toks.grad.detach()
        all_grad_w.append(grad.norm(dim=-1).cpu().numpy())
        model.zero_grad()

    grad_w = np.concatenate(all_grad_w)
    # Return normalized average profile
    profile = grad_w.mean(axis=0)
    profile = profile / (profile.sum() + 1e-8)
    return profile


# ─────────────────────────────────────────────────────────────
# CLASSICAL GCC METHODS
# ─────────────────────────────────────────────────────────────

def gcc_tdoa(mic1, mic2, tau_max, method='phat', learned_w=None):
    """
    Estimate TDOA using GCC with specified weighting.

    Args:
        mic1, mic2: time-domain signals (T,)
        tau_max: maximum |tau| to search
        method: 'phat', 'magnitude', 'scot', 'flat', 'learned'
        learned_w: frequency weighting profile for method='learned', shape (F,)

    Returns:
        estimated tau (float, in samples)
    """
    X1 = np.fft.rfft(mic1)
    X2 = np.fft.rfft(mic2)
    G12 = X2 * np.conj(X1)  # X2·X1* so IFFT peak is at +τ (not -τ)
    eps = 1e-10

    if method == 'phat':
        W = 1.0 / (np.abs(G12) + eps)
    elif method == 'magnitude':
        W = np.abs(G12)
    elif method == 'scot':
        W = 1.0 / (np.sqrt(np.abs(X1)**2 * np.abs(X2)**2) + eps)
    elif method == 'flat':
        W = np.ones_like(G12, dtype=np.float64)
    elif method == 'learned':
        W = learned_w.astype(np.float64)
    else:
        raise ValueError(f"Unknown method: {method}")

    R = np.fft.irfft(W * G12, len(mic1))

    # Search within [-tau_max, +tau_max]
    T = len(mic1)
    candidates = np.concatenate([
        np.arange(0, tau_max + 1),
        np.arange(T - tau_max, T)
    ])
    best_idx = candidates[np.argmax(R[candidates])]
    tau_est = best_idx if best_idx <= tau_max else best_idx - T
    return float(tau_est)


def evaluate_gcc_methods(n_test, snr_db, noise_type, learned_w=None, rng_seed=777):
    """
    Evaluate all GCC methods on fresh test data.
    Returns dict of {method: MAE in samples}.
    """
    rng = np.random.RandomState(rng_seed)
    methods = ['phat', 'scot', 'flat', 'magnitude']
    if learned_w is not None:
        methods.append('learned')

    results = {m: [] for m in methods}
    taus_true = []

    for _ in tqdm(range(n_test), desc='  GCC eval', leave=False):
        tau_true = rng.uniform(-TAU_MAX, TAU_MAX)
        taus_true.append(tau_true)
        mic1, mic2 = make_pair(tau_true, snr_db, T, noise_type, rng)

        for m in methods:
            tau_est = gcc_tdoa(mic1, mic2, TAU_MAX, method=m,
                               learned_w=learned_w if m == 'learned' else None)
            results[m].append(abs(tau_est - tau_true))

    return {m: float(np.mean(v)) for m, v in results.items()}



# ─────────────────────────────────────────────────────────────
# ROOM EVALUATION (pyroomacoustics)
# ─────────────────────────────────────────────────────────────

def evaluate_room_gcc(n_test, t60_range, learned_w, rng_seed=777):
    """Evaluate GCC methods under reverberant conditions."""
    try:
        import pyroomacoustics as pra
    except ImportError:
        print("  pyroomacoustics not available, skipping room eval")
        return None

    FS = 16000
    MIC_SEP = 0.5
    SNR_DB = 10
    rng = np.random.RandomState(rng_seed)
    methods = ['phat', 'scot', 'flat', 'magnitude', 'learned']
    results = {m: [] for m in methods}

    for _ in tqdm(range(n_test), desc=f'  Room T60={t60_range}', leave=False):
        t60 = rng.uniform(*t60_range)
        room_dim = rng.uniform(3, 8, size=3)  # smaller rooms to avoid inverse_sabine failure
        room_dim[2] = rng.uniform(2.5, 3.5)

        try:
            e_abs, max_order = pra.inverse_sabine(t60, room_dim)
        except ValueError:
            continue  # room too large for this T60
        if max_order > 50: max_order = 50
        room = pra.ShoeBox(room_dim, fs=FS, materials=pra.Material(e_abs),
                           max_order=max_order)

        mic_center = rng.uniform(1, room_dim - 1, size=3)
        mic_center[2] = np.clip(mic_center[2], 1.0, room_dim[2] - 0.5)
        mic1_pos = mic_center.copy(); mic1_pos[0] -= MIC_SEP / 2
        mic2_pos = mic_center.copy(); mic2_pos[0] += MIC_SEP / 2

        if np.any(mic1_pos < 0.1) or np.any(mic2_pos < 0.1):
            continue
        if np.any(mic1_pos > room_dim - 0.1) or np.any(mic2_pos > room_dim - 0.1):
            continue

        room.add_microphone_array(np.array([mic1_pos, mic2_pos]).T)
        src_pos = rng.uniform(1, room_dim - 1, size=3)
        src_pos[2] = np.clip(src_pos[2], 1.0, room_dim[2] - 0.5)
        room.add_source(src_pos)

        room.compute_rir()
        rir1, rir2 = room.rir[0][0], room.rir[1][0]

        # True TDOA from direct path geometry
        d1 = np.linalg.norm(src_pos - mic1_pos)
        d2 = np.linalg.norm(src_pos - mic2_pos)
        tau_true_sec = (d1 - d2) / 343.0
        tau_true_samples = tau_true_sec * FS

        max_tau_samples = int(MIC_SEP / 343.0 * FS) + 2
        if abs(tau_true_samples) > max_tau_samples:
            continue

        # Generate reverberant signals
        src_signal = rng.randn(FS).astype(np.float32)
        sig1 = np.convolve(src_signal, rir1)[:FS]
        sig2 = np.convolve(src_signal, rir2)[:FS]

        # Add noise
        ns = 10**(-SNR_DB/20)
        sig1 += rng.randn(len(sig1)).astype(np.float32) * ns * np.std(sig1)
        sig2 += rng.randn(len(sig2)).astype(np.float32) * ns * np.std(sig2)

        # Take a 256-sample segment from the middle
        start = FS // 2
        seg1 = sig1[start:start+T].astype(np.float64)
        seg2 = sig2[start:start+T].astype(np.float64)

        # Scale tau_max for room case (in 256-sample frame at 16kHz)
        room_tau_max = max_tau_samples + 2

        for m in methods:
            tau_est = gcc_tdoa(seg1, seg2, room_tau_max, method=m,
                               learned_w=learned_w if m == 'learned' else None)
            results[m].append(abs(tau_est - tau_true_samples))

    if all(len(v) == 0 for v in results.values()):
        return None

    return {m: float(np.mean(v)) if v else float('nan') for m, v in results.items()}


# ─────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    N_TRAIN, N_VAL = 50_000, 5_000
    N_TEST_GCC     = 10_000
    EPOCHS         = 120
    SNR_LEVELS     = [20, 10, 5, 0, -5, -10]
    NOISE_TYPES    = ['white', 'colored']

    all_results = {}

    # ── Part 1: SNR sweep comparison ──
    print(f"\n{'━'*70}")
    print("  PART 1: GCC-Learned vs GCC-PHAT — SNR Sweep")
    print(f"{'━'*70}")

    for noise_type in NOISE_TYPES:
        for snr_db in SNR_LEVELS:
            tag = f"{noise_type}_snr{snr_db:+d}dB"
            print(f"\n{'─'*50}")
            print(f"  {tag}")
            print(f"{'─'*50}")

            # 1. Train Transformer
            print("  [1] Training Transformer ...")
            tr_ds = TDOADataset(N_TRAIN, TAU_MAX, float(snr_db), T, noise_type, seed=42)
            va_ds = TDOADataset(N_VAL,   TAU_MAX, float(snr_db), T, noise_type, seed=99)
            model = TransformerModel(F=F)
            train_model(model, tr_ds, va_ds, epochs=EPOCHS)

            # 2. Extract learned weighting profile
            print("  [2] Extracting learned weighting ...")
            learned_w = extract_gradient_profile(model, va_ds, n_samples=2000)
            print(f"       Profile peak at bin {np.argmax(learned_w)}, "
                  f"top-10 energy: {np.sort(learned_w)[-10:].sum():.3f}")

            # 3. Evaluate all GCC methods
            print("  [3] Evaluating GCC methods ...")
            gcc_results = evaluate_gcc_methods(
                N_TEST_GCC, float(snr_db), noise_type, learned_w, rng_seed=777)

            # Store
            all_results[tag] = {
                'gcc': gcc_results,
                'learned_profile': learned_w,
            }

            # Print comparison
            print(f"       Results (MAE in samples):")
            print(f"       {'Method':15s}  {'MAE':>8s}")
            for m in ['phat', 'flat', 'scot', 'magnitude', 'learned']:
                marker = ' ◀ BEST' if m == min(gcc_results, key=gcc_results.get) else ''
                print(f"       {('GCC-'+m.upper()):15s}  {gcc_results[m]:>8.3f}{marker}")

    # ── Part 2: Room acoustics comparison ──
    print(f"\n{'━'*70}")
    print("  PART 2: GCC-Learned vs GCC-PHAT — Room Acoustics")
    print(f"{'━'*70}")

    # Use the learned profile from colored 0dB (representative condition)
    baseline_profile = all_results['colored_snr+0dB']['learned_profile']

    room_results = {}
    for t60_label, t60_range in [
        ('T60=0.2', (0.15, 0.25)),
        ('T60=0.4', (0.35, 0.45)),
        ('T60=0.6', (0.55, 0.70)),
    ]:
        print(f"\n  {t60_label} ...")
        res = evaluate_room_gcc(2000, t60_range, baseline_profile, rng_seed=888)
        if res is not None:
            room_results[t60_label] = res
            print(f"       {'Method':15s}  {'MAE':>8s}")
            for m in ['phat', 'flat', 'scot', 'magnitude', 'learned']:
                marker = ' ◀ BEST' if m == min(res, key=res.get) else ''
                print(f"       {('GCC-'+m.upper()):15s}  {res[m]:>8.3f}{marker}")

    all_results['room'] = room_results

    # ── Save ──
    torch.save(all_results, 'results/gcc_benchmark.pt')

    # ── Part 3: Plots ──
    print(f"\n{'━'*70}")
    print("  PLOTTING")
    print(f"{'━'*70}")

    # --- Plot 1: Main comparison (2 panels: white + colored) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    method_colors = {
        'phat': 'royalblue', 'flat': 'gray', 'scot': 'green',
        'magnitude': 'darkorange', 'learned': 'crimson',
    }
    method_labels = {
        'phat': 'GCC-PHAT', 'flat': 'GCC-Flat', 'scot': 'GCC-SCOT',
        'magnitude': 'GCC-Mag', 'learned': 'GCC-Learned',
    }

    for col, noise_type in enumerate(NOISE_TYPES):
        ax = axes[col]
        for m in ['phat', 'flat', 'scot', 'magnitude', 'learned']:
            maes = [all_results[f'{noise_type}_snr{s:+d}dB']['gcc'][m]
                    for s in SNR_LEVELS]
            ax.plot(SNR_LEVELS, maes, 'o-', color=method_colors[m],
                    lw=2.5, ms=8, label=method_labels[m],
                    zorder=5 if m == 'learned' else 3)

        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel('MAE (samples)', fontsize=11)
        ax.set_title(f'{noise_type.title()} Noise', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle(
        'GCC-Learned vs Classical GCC Methods — TDOA Estimation Accuracy\n'
        'Learned weighting extracted from trained Transformer via gradient attribution',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/gcc_benchmark_snr.png', dpi=150, bbox_inches='tight')
    print("  → Saved: results/gcc_benchmark_snr.png")
    plt.close()

    # --- Plot 2: Room acoustics bar chart ---
    if room_results:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        conditions = list(room_results.keys())
        methods_plot = ['phat', 'flat', 'scot', 'magnitude', 'learned']
        n_methods = len(methods_plot)
        x = np.arange(len(conditions))
        width = 0.15

        for i, m in enumerate(methods_plot):
            vals = [room_results[c].get(m, float('nan')) for c in conditions]
            ax.bar(x + (i - n_methods/2 + 0.5) * width, vals, width,
                   color=method_colors[m], label=method_labels[m], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=11)
        ax.set_ylabel('MAE (samples @ 16kHz)', fontsize=11)
        ax.set_title('GCC-Learned vs Classical — Room Acoustics (pyroomacoustics)\n'
                      'Learned weighting from colored 0dB Transformer',
                      fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('results/gcc_benchmark_room.png', dpi=150, bbox_inches='tight')
        print("  → Saved: results/gcc_benchmark_room.png")
        plt.close()

    # --- Plot 3: Compact summary — improvement over PHAT ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for col, noise_type in enumerate(NOISE_TYPES):
        ax = axes[col]
        phat_maes = [all_results[f'{noise_type}_snr{s:+d}dB']['gcc']['phat']
                     for s in SNR_LEVELS]
        for m in ['magnitude', 'learned']:
            m_maes = [all_results[f'{noise_type}_snr{s:+d}dB']['gcc'][m]
                      for s in SNR_LEVELS]
            improvement = [(p - l) / p * 100 for p, l in zip(phat_maes, m_maes)]
            ax.plot(SNR_LEVELS, improvement, 'o-', color=method_colors[m],
                    lw=2.5, ms=8, label=method_labels[m])

        ax.axhline(0, color='gray', lw=1, ls=':')
        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel('Improvement over GCC-PHAT (%)', fontsize=11)
        ax.set_title(f'{noise_type.title()} Noise', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.suptitle(
        'Relative Improvement over GCC-PHAT\n'
        '(positive = better than PHAT)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/gcc_benchmark_improvement.png', dpi=150, bbox_inches='tight')
    print("  → Saved: results/gcc_benchmark_improvement.png")
    plt.close()

    # ── Summary Table ──
    print(f"\n{'━'*70}")
    print("GCC BENCHMARK SUMMARY")
    print(f"{'━'*70}")

    print(f"\n  SNR Sweep (MAE in samples, lower is better):")
    print(f"  {'Condition':20s}  {'GCC-PHAT':>9s}  {'GCC-Flat':>9s}  {'GCC-Mag':>9s}"
          f"  {'GCC-Learn':>9s}  {'Learn vs PHAT':>14s}")

    for noise_type in NOISE_TYPES:
        for snr_db in SNR_LEVELS:
            tag = f"{noise_type}_snr{snr_db:+d}dB"
            r = all_results[tag]
            g = r['gcc']
            phat_mae = g['phat']
            learned_mae = g['learned']
            delta = (phat_mae - learned_mae) / phat_mae * 100 if phat_mae > 0 else 0
            sign = '+' if delta > 0 else ''
            print(f"  {tag:20s}  {phat_mae:>9.3f}  {g['flat']:>9.3f}  "
                  f"{g['magnitude']:>9.3f}  {learned_mae:>9.3f}  "
                  f"{sign}{delta:>12.1f}%")

    if room_results:
        print(f"\n  Room Acoustics (MAE in samples @ 16kHz):")
        print(f"  {'Condition':20s}  {'GCC-PHAT':>9s}  {'GCC-Flat':>9s}  {'GCC-Mag':>9s}"
              f"  {'GCC-Learn':>9s}  {'Learn vs PHAT':>14s}")
        for cond, res in room_results.items():
            phat_mae = res['phat']
            learned_mae = res['learned']
            delta = (phat_mae - learned_mae) / phat_mae * 100 if phat_mae > 0 else 0
            sign = '+' if delta > 0 else ''
            print(f"  {cond:20s}  {phat_mae:>9.3f}  {res['flat']:>9.3f}  "
                  f"{res['magnitude']:>9.3f}  {learned_mae:>9.3f}  "
                  f"{sign}{delta:>12.1f}%")

    print(f"\nFigures: results/gcc_benchmark_*.png")
    print(f"\n{'━'*70}")
    print("DONE")
    print(f"{'━'*70}")
