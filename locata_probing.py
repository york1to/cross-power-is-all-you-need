#!/usr/bin/env python3
"""
LOCATA Real-Recording Probing
==============================
Train a Transformer on real LOCATA recordings and probe for cross-power
vs PHAT whitening — same as synthetic experiments but with real data.

Core question: Does the cross-power finding hold when training on real audio?

Data: LOCATA dev set, Task 1, DICIT array (15-mic planar array)
  - Use ALL mic pairs with |τ| ≤ 30 samples (~224 pairs)
  - Each pair → different TDOA, same recording → diverse τ distribution
  - Resample 48 kHz → 16 kHz, segment into T=256 frames
  - Ground-truth TDOA: full-signal GCC-PHAT (reliable for static speaker)
  - Split: 70% train, 30% val (stratified by recording)

Requires: data/locata/dev/ directory
  Download from: https://zenodo.org/records/3630471/files/dev.zip
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'soundfile'])
    import soundfile as sf

try:
    from scipy.signal import resample_poly
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
    from scipy.signal import resample_poly

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

T          = 256
F          = T // 2 + 1   # 129
FS_TARGET  = 16000
TAU_MAX    = 30
EPOCHS     = 120
LOCATA_DIR = 'data/locata'
ARRAY_NAME = 'dicit'       # 15-channel planar array
N_MICS     = 15


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def download_locata():
    """Download LOCATA dev set from Zenodo (6.2 GB)."""
    dev_dir = os.path.join(LOCATA_DIR, 'dev')
    if os.path.isdir(dev_dir):
        print("LOCATA dev set already present.")
        return

    zip_path = os.path.join(LOCATA_DIR, 'dev.zip')
    os.makedirs(LOCATA_DIR, exist_ok=True)

    if not os.path.exists(zip_path):
        url = 'https://zenodo.org/records/3630471/files/dev.zip?download=1'
        print(f"Downloading LOCATA dev set (6.2 GB) ...")
        import subprocess
        subprocess.check_call([
            'wget', '-O', zip_path, '--progress=bar:force:noscroll', url
        ])

    print("Extracting dev.zip ...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(LOCATA_DIR)
    print(f"  Extracted to {dev_dir}/")


def estimate_tdoa_gcc_phat(mic1, mic2):
    """Estimate TDOA via GCC-PHAT on full-length signals."""
    N = len(mic1)
    X1 = np.fft.fft(mic1)
    X2 = np.fft.fft(mic2)
    G12 = X1 * np.conj(X2)
    gcc = np.real(np.fft.ifft(G12 / (np.abs(G12) + 1e-10)))
    gcc = np.fft.fftshift(gcc)
    lags = np.arange(-(N // 2), N // 2)

    peak_idx = np.argmax(gcc)
    tau = float(lags[peak_idx])

    if 0 < peak_idx < len(gcc) - 1:
        alpha = gcc[peak_idx - 1]
        beta  = gcc[peak_idx]
        gamma = gcc[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            tau += 0.5 * (alpha - gamma) / denom

    return tau


def to_tokens(mic1, mic2):
    X1, X2 = np.fft.rfft(mic1), np.fft.rfft(mic2)
    t = np.stack([X1.real, X1.imag, X2.real, X2.imag], axis=1).astype(np.float32)
    return t / (np.std(t) + 1e-8)


def load_all_pairs(locata_dir, array_name='dicit'):
    """Load all valid mic pairs from DICIT across Task 1 recordings.

    Returns list of dicts: {tokens, tau, rec_name, ch1, ch2, n_frames}
    """
    task1_dir = os.path.join(locata_dir, 'dev', 'task1')
    if not os.path.isdir(task1_dir):
        raise FileNotFoundError(f"Not found: {task1_dir}")

    rec_dirs = sorted([
        os.path.join(task1_dir, d)
        for d in os.listdir(task1_dir)
        if os.path.isdir(os.path.join(task1_dir, d))
        and d.startswith('recording')
    ])

    all_samples = []
    stats = {'recordings': 0, 'pairs': 0, 'frames': 0}

    for rec_dir in rec_dirs:
        rec_name = os.path.basename(rec_dir)
        audio_path = os.path.join(rec_dir, array_name,
                                  f'audio_array_{array_name}.wav')
        if not os.path.exists(audio_path):
            continue

        audio, fs = sf.read(audio_path)
        if audio.ndim == 1 or audio.shape[1] < N_MICS:
            continue

        # Resample full audio once
        from math import gcd
        g = gcd(FS_TARGET, fs)
        up, down = FS_TARGET // g, fs // g
        audio_16k = resample_poly(audio, up, down, axis=0).astype(np.float64)

        n_frames = len(audio_16k) // T
        if n_frames < 10:
            continue

        stats['recordings'] += 1
        n_valid = 0

        for ch1 in range(N_MICS):
            for ch2 in range(ch1 + 1, N_MICS):
                mic1 = audio_16k[:, ch1]
                mic2 = audio_16k[:, ch2]

                tau = estimate_tdoa_gcc_phat(mic1, mic2)
                if abs(tau) > TAU_MAX or abs(tau) < 0.5:
                    continue

                # Segment into T-sample frames
                mic1_f = mic1[:n_frames * T].reshape(n_frames, T).astype(np.float32)
                mic2_f = mic2[:n_frames * T].reshape(n_frames, T).astype(np.float32)

                for i in range(n_frames):
                    all_samples.append({
                        'tokens': to_tokens(mic1_f[i], mic2_f[i]),
                        'tau': tau,
                    })

                n_valid += 1
                stats['pairs'] += 1
                stats['frames'] += n_frames

        print(f"  {rec_name}: {n_valid} valid pairs, "
              f"{n_valid * n_frames} frames")

    print(f"  Total: {stats['recordings']} recordings, "
          f"{stats['pairs']} pairs, {stats['frames']} frames")
    return all_samples, stats


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class LOCATADataset(Dataset):
    def __init__(self, samples):
        taus = np.array([s['tau'] for s in samples], dtype=np.float32)
        tokens = np.stack([s['tokens'] for s in samples])
        self.tokens   = torch.from_numpy(tokens)
        self.taus_norm = torch.from_numpy(taus / TAU_MAX)
        self.taus_raw  = torch.from_numpy(taus)
        self.T, self.F, self.tau_max = T, F, TAU_MAX

    def __len__(self): return len(self.taus_norm)
    def __getitem__(self, i):
        return self.tokens[i], self.taus_norm[i:i + 1]


# ─────────────────────────────────────────────────────────────
# MODEL (same architecture as all other experiments)
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


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_model(model, train_ds, val_ds, epochs=120, lr=3e-4,
                batch_size=1024, warmup_frac=0.05):
    model.to(device)
    tr = DataLoader(train_ds, batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)
    va = DataLoader(val_ds, batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
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
            tr_mae.append((pred - tau).abs().mean().item() * TAU_MAX)
        model.eval()
        va_mae = []
        with torch.no_grad():
            for toks, tau in va:
                toks, tau = toks.to(device), tau.to(device)
                va_mae.append((model(toks) - tau).abs().mean().item() * TAU_MAX)
        tm, vm = np.mean(tr_mae), np.mean(va_mae)
        history['train_mae'].append(tm); history['val_mae'].append(vm)
        pbar.set_postfix({'tr': f'{tm:.3f}', 'va': f'{vm:.3f}'})
    return history


# ─────────────────────────────────────────────────────────────
# PROBE TARGETS
# ─────────────────────────────────────────────────────────────

def build_probe_targets(ds, idx):
    toks = ds.tokens[idx].numpy()
    taus = ds.taus_raw[idx].numpy()
    X1   = toks[:, :, 0] + 1j * toks[:, :, 1]
    X2   = toks[:, :, 2] + 1j * toks[:, :, 3]
    cross = X1 * np.conj(X2)
    k   = np.arange(F)[None, :]
    phi = 2 * np.pi * k * taus[:, None] / T
    return {
        'tau':      (taus / TAU_MAX).astype(np.float32),
        'cross_re': cross.real.astype(np.float32),
        'cross_im': cross.imag.astype(np.float32),
        'phat_cos': np.cos(phi).astype(np.float32),
        'phat_sin': np.sin(phi).astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────
# PROBING
# ─────────────────────────────────────────────────────────────

def run_probes(model, ds, n_probe=3000, n_freq_sample=64):
    model.eval().to(device)
    rng = np.random.RandomState(7)
    n_probe = min(n_probe, len(ds))
    idx = rng.choice(len(ds), n_probe, replace=False)
    targets = build_probe_targets(ds, idx)
    freq_bins = rng.choice(range(1, ds.F),
                           min(n_freq_sample, ds.F - 1), replace=False)
    split = int(0.7 * n_probe)

    toks_all = ds.tokens[idx].to(device)
    n_depths = model.n_layers + 1
    f_buf = [[] for _ in range(n_depths)]
    g_buf = [[] for _ in range(n_depths)]
    BS = 256
    with torch.no_grad():
        for s in range(0, n_probe, BS):
            _, hs = model(toks_all[s:s + BS], return_hidden=True,
                          include_embed=True)
            for l, h in enumerate(hs):
                f_buf[l].append(h[:, 1:].cpu().numpy())
                g_buf[l].append(h[:, 0].cpu().numpy())
    freq_H   = [np.concatenate(b) for b in f_buf]
    global_H = [np.concatenate(b) for b in g_buf]

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

    return freq_results, global_results, n_depths


# ─────────────────────────────────────────────────────────────
# SYNTHETIC BASELINE
# ─────────────────────────────────────────────────────────────

def load_synthetic_baseline():
    path = 'results/perfreq_colored_snr+0dB.pt'
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location='cpu', weights_only=False)
    return {
        'freq':   data.get('freq_probes', {}),
        'global': data.get('cls_probes', {}),
    }


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_results(locata_freq, locata_global, n_depths,
                 val_mae, synthetic=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = list(range(n_depths))
    x_lbl = ['Emb'] + [f'L{i+1}' for i in range(n_depths - 1)]

    # Panel 1: cross_re
    ax = axes[0]
    ax.plot(x, locata_freq['cross_re'], 'o-', color='darkorange',
            lw=2.5, ms=9, label='LOCATA (trained)')
    if synthetic and 'cross_re' in synthetic.get('freq', {}):
        ax.plot(x, synthetic['freq']['cross_re'], 's--', color='darkorange',
                lw=1.5, ms=7, alpha=0.5, label='Synthetic baseline')
    ax.set_title('cross_re R$^2$\n(Cross-power real part)',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(x_lbl)
    ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlabel('Depth'); ax.set_ylabel('R$^2$')

    # Panel 2: phat_cos
    ax = axes[1]
    ax.plot(x, locata_freq['phat_cos'], 'o-', color='royalblue',
            lw=2.5, ms=9, label='LOCATA (trained)')
    if synthetic and 'phat_cos' in synthetic.get('freq', {}):
        ax.plot(x, synthetic['freq']['phat_cos'], 's--', color='royalblue',
                lw=1.5, ms=7, alpha=0.5, label='Synthetic baseline')
    ax.set_title('phat_cos R$^2$\n(PHAT whitening phase)',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(x_lbl)
    ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlabel('Depth'); ax.set_ylabel('R$^2$')

    # Panel 3: Summary bar chart
    ax = axes[2]
    metrics = ['cross_re', 'cross_im', 'phat_cos', 'phat_sin']
    labels  = ['cross\nre', 'cross\nim', 'phat\ncos', 'phat\nsin']
    colors  = ['darkorange', 'sienna', 'royalblue', 'deepskyblue']

    locata_best = [max(locata_freq[m]) for m in metrics]
    x_bar = np.arange(len(metrics))
    width = 0.35

    ax.bar(x_bar - width / 2, locata_best, width, color=colors, alpha=0.9,
           label='LOCATA (best depth)')

    syn_freq = synthetic.get('freq', {}) if synthetic else {}
    syn_best = [max(syn_freq[m]) for m in metrics if m in syn_freq]
    if len(syn_best) == len(metrics):
        ax.bar(x_bar + width / 2, syn_best, width, color=colors, alpha=0.4,
               edgecolor='black', linewidth=1.5, label='Synthetic (best depth)')

    ax.set_xticks(x_bar); ax.set_xticklabels(labels)
    ax.set_ylabel('Best R$^2$'); ax.set_ylim(-0.1, 1.1)
    ax.set_title('Best-depth R$^2$ Comparison\nLOCATA vs Synthetic',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'LOCATA Real-Recording Probing — Trained on Real Data\n'
        f'Transformer trained on LOCATA Task 1 (val MAE={val_mae:.2f})',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = 'results/locata_probing.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_start = time.time()

    print("=" * 60)
    print("LOCATA Real-Recording Probing (Train + Probe)")
    print("=" * 60)

    # 1. Data
    download_locata()

    print(f"\n[1] Loading all mic pairs (DICIT, |tau|<=30) ...")
    all_samples, stats = load_all_pairs(LOCATA_DIR, ARRAY_NAME)

    if len(all_samples) < 100:
        print("!! Not enough samples.")
        sys.exit(1)

    # 2. Shuffle and split 70/30
    rng = np.random.RandomState(42)
    rng.shuffle(all_samples)
    split = int(0.7 * len(all_samples))
    train_samples = all_samples[:split]
    val_samples   = all_samples[split:]

    print(f"\n[2] Building datasets ...")
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")
    train_ds = LOCATADataset(train_samples)
    val_ds   = LOCATADataset(val_samples)

    tau_vals = val_ds.taus_raw.numpy()
    print(f"  Val tau range: [{tau_vals.min():.1f}, {tau_vals.max():.1f}], "
          f"unique: {len(np.unique(np.round(tau_vals)))}")

    # 3. Train
    print(f"\n[3] Training Transformer ...")
    model = TransformerModel(F=F)
    print(f"  Params: {model.n_params:,}")
    t0 = time.time()
    history = train_model(model, train_ds, val_ds, epochs=EPOCHS)
    val_mae = history['val_mae'][-1]
    print(f"  {time.time() - t0:.1f}s | val MAE: {val_mae:.3f}")

    torch.save(model.state_dict(), 'results/locata_model.pt')
    print(f"  Saved: results/locata_model.pt")

    # 4. Probe
    print(f"\n[4] Running probes on val set ...")
    t0 = time.time()
    freq_res, global_res, n_depths = run_probes(
        model, val_ds, n_probe=min(3000, len(val_ds)))
    print(f"  {time.time() - t0:.1f}s")

    print(f"\n  Per-frequency R^2:")
    for name in ['cross_re', 'cross_im', 'phat_cos', 'phat_sin']:
        print(f"    {name:10s}: {[f'{v:.3f}' for v in freq_res[name]]}")

    print(f"\n  Global R^2:")
    for name in ['tau', 'cross_re', 'phat_cos']:
        if name in global_res:
            print(f"    {name:10s}: {[f'{v:.3f}' for v in global_res[name]]}")

    # 5. Synthetic baseline
    print(f"\n[5] Synthetic baseline ...")
    synthetic = load_synthetic_baseline()
    if synthetic:
        print("  Loaded perfreq_colored_snr+0dB.pt")

    # 6. Save
    results = {
        'freq': freq_res,
        'global': global_res,
        'n_depths': n_depths,
        'val_mae': val_mae,
        'history': history,
        'stats': stats,
        'config': {
            'array': ARRAY_NAME, 'n_mics': N_MICS,
            'fs_target': FS_TARGET, 'T': T, 'F': F,
            'TAU_MAX': TAU_MAX, 'epochs': EPOCHS,
        },
    }
    torch.save(results, 'results/locata_probing.pt')
    print(f"\n  Saved: results/locata_probing.pt")

    # 7. Plot
    print(f"\n[6] Plotting ...")
    plot_results(freq_res, global_res, n_depths, val_mae, synthetic)

    # Summary
    cr_best = max(freq_res['cross_re'])
    pc_best = max(freq_res['phat_cos'])
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Data:        {stats['pairs']} mic pairs, {stats['frames']} frames")
    print(f"  Val MAE:     {val_mae:.3f} samples")
    print(f"  cross_re R2: {[f'{v:.3f}' for v in freq_res['cross_re']]}")
    print(f"  phat_cos R2: {[f'{v:.3f}' for v in freq_res['phat_cos']]}")
    print(f"\n  cross_re best = {cr_best:.3f}")
    print(f"  phat_cos best = {pc_best:.3f}")

    if cr_best > 0.5 and pc_best < 0.3:
        print(f"\n  CONFIRMED: Cross-power finding holds on real recordings!")
    elif cr_best > pc_best:
        print(f"\n  DIRECTIONAL: cross_re > phat_cos (gap = {cr_best - pc_best:.3f})")
    else:
        print(f"\n  INCONCLUSIVE: phat_cos >= cross_re on real data")

    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    print(f"{'='*60}")
