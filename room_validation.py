#!/usr/bin/env python3
"""
Room Validation — Real Acoustic Environment with pyroomacoustics
================================================================
Core question: Do synthetic findings hold under reverberation?

If cross_re R² stays high and phat_cos R² stays low with real RIRs,
the finding is not an artifact of synthetic data.

Conditions:
  T60 ∈ {0.2, 0.4, 0.6} + mixed (0.15–0.7), all at SNR=10 dB
  Random rooms (3–10m per dim), 2-mic pair (0.5m separation)
  Source: white noise convolved with room impulse response

Dataset: 500 rooms/condition → 30k train, 3k val per condition
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

try:
    import pyroomacoustics as pra
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyroomacoustics'])
    import pyroomacoustics as pra

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

T        = 256
F        = T // 2 + 1   # 129
FS       = 16000         # 16 kHz sampling rate
C        = 343.0         # speed of sound
MIC_SEP  = 0.5           # mic pair separation (m)
TAU_MAX  = 30            # max TDOA in samples (0.5m/343*16000 ≈ 23.3)
SNR_DB   = 10.0
EPOCHS   = 120
N_TRAIN  = 30_000
N_VAL    = 3_000

T60_CONDITIONS = {
    't60_0.2': (0.15, 0.25),
    't60_0.4': (0.35, 0.45),
    't60_0.6': (0.55, 0.70),
    'mixed':   (0.15, 0.70),
}


# ─────────────────────────────────────────────────────────────
# RIR GENERATION
# ─────────────────────────────────────────────────────────────

def generate_rir_pool(n_rooms, t60_range, rng):
    """Generate a pool of room impulse response pairs with known TDOAs."""
    pool = []
    for _ in tqdm(range(n_rooms), desc='  RIRs', leave=False):
        room_x = rng.uniform(3, 10)
        room_y = rng.uniform(3, 10)
        room_z = rng.uniform(2.5, 4.0)
        t60 = rng.uniform(t60_range[0], t60_range[1])

        try:
            e_abs, max_order = pra.inverse_sabine(t60, [room_x, room_y, room_z])
        except ValueError:
            continue
        max_order = min(int(max_order), 30)

        room = pra.ShoeBox(
            [room_x, room_y, room_z], fs=FS,
            materials=pra.Material(e_abs), max_order=max_order)

        # Mic pair at random position
        mc = np.array([
            room_x * rng.uniform(0.3, 0.7),
            room_y * rng.uniform(0.3, 0.7),
            rng.uniform(1.2, 1.8)])
        mic_pos = np.array([
            [mc[0] - MIC_SEP / 2, mc[1], mc[2]],
            [mc[0] + MIC_SEP / 2, mc[1], mc[2]],
        ]).T                                         # (3, 2)
        room.add_microphone_array(mic_pos)

        # Random source
        src_pos = np.array([
            rng.uniform(0.5, room_x - 0.5),
            rng.uniform(0.5, room_y - 0.5),
            rng.uniform(1.0, 2.0)])
        room.add_source(src_pos)
        room.compute_rir()

        # Direct-path TDOA
        d1 = np.linalg.norm(src_pos - mic_pos[:, 0])
        d2 = np.linalg.norm(src_pos - mic_pos[:, 1])
        tau = (d2 - d1) / C * FS

        if abs(tau) > TAU_MAX:
            continue

        pool.append({
            'rir1': room.rir[0][0].astype(np.float32),
            'rir2': room.rir[1][0].astype(np.float32),
            'tau':  float(tau),
            't60':  t60,
        })
    return pool


def make_room_pair(rir_entry, T, snr_db, rng):
    """Convolve random source with RIR pair → mic signals + tokens."""
    rir1, rir2 = rir_entry['rir1'], rir_entry['rir2']
    rir_len = max(len(rir1), len(rir2))

    # Generate source (longer than needed for RIR convolution)
    src = rng.randn(T + rir_len).astype(np.float32)

    # Convolve
    mic1 = np.convolve(src, rir1, mode='full').astype(np.float32)
    mic2 = np.convolve(src, rir2, mode='full').astype(np.float32)

    # Take T samples after RIR has built up
    start = rir_len
    mic1 = mic1[start:start + T]
    mic2 = mic2[start:start + T]

    if len(mic1) < T or len(mic2) < T:
        return None

    # Normalize
    mic1 /= np.std(mic1) + 1e-8
    mic2 /= np.std(mic2) + 1e-8

    # Add noise
    if not np.isinf(snr_db):
        ns = 10 ** (-snr_db / 20)
        mic1 = mic1 + rng.randn(T).astype(np.float32) * ns
        mic2 = mic2 + rng.randn(T).astype(np.float32) * ns

    return mic1, mic2


def to_tokens(mic1, mic2):
    X1, X2 = np.fft.rfft(mic1), np.fft.rfft(mic2)
    t = np.stack([X1.real, X1.imag, X2.real, X2.imag], axis=1).astype(np.float32)
    return t / (np.std(t) + 1e-8)


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class RoomTDOADataset(Dataset):
    def __init__(self, rir_pool, n_samples, snr_db, T, seed=0):
        rng = np.random.RandomState(seed)
        taus, toks_list = [], []
        for _ in tqdm(range(n_samples), desc='  data', leave=False):
            entry = rir_pool[rng.randint(len(rir_pool))]
            result = make_room_pair(entry, T, snr_db, rng)
            if result is None:
                continue
            mic1, mic2 = result
            toks_list.append(to_tokens(mic1, mic2))
            taus.append(entry['tau'])

        taus = np.array(taus, dtype=np.float32)
        self.tokens    = torch.from_numpy(np.stack(toks_list))
        self.taus_norm = torch.from_numpy(taus / TAU_MAX)
        self.taus_raw  = torch.from_numpy(taus)
        self.T, self.F, self.tau_max = T, F, TAU_MAX

    def __len__(self): return len(self.taus_norm)
    def __getitem__(self, i):
        return self.tokens[i], self.taus_norm[i:i + 1]


# ─────────────────────────────────────────────────────────────
# MODEL (Transformer — same architecture as all other experiments)
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
# PROBE TARGETS (adapted for room data)
# ─────────────────────────────────────────────────────────────

def build_probe_targets(ds, idx):
    """Same as perfreq_probe.py but standalone."""
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

def run_probes(model, val_ds, n_probe=3000, n_freq_sample=64):
    model.eval().to(device)
    rng = np.random.RandomState(7)
    n_probe = min(n_probe, len(val_ds))
    idx = rng.choice(len(val_ds), n_probe, replace=False)
    targets = build_probe_targets(val_ds, idx)
    freq_bins = rng.choice(range(1, val_ds.F),
                           min(n_freq_sample, val_ds.F - 1), replace=False)
    split = int(0.7 * n_probe)

    toks_all = val_ds.tokens[idx].to(device)
    n_depths = model.n_layers + 1
    f_buf = [[] for _ in range(n_depths)]
    g_buf = [[] for _ in range(n_depths)]
    BS = 256
    with torch.no_grad():
        for s in range(0, n_probe, BS):
            _, hs = model(toks_all[s:s + BS], return_hidden=True, include_embed=True)
            for l, h in enumerate(hs):
                f_buf[l].append(h[:, 1:].cpu().numpy())
                g_buf[l].append(h[:, 0].cpu().numpy())
    freq_H   = [np.concatenate(b) for b in f_buf]
    global_H = [np.concatenate(b) for b in g_buf]

    # Per-frequency probes
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

    # Global probes
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
# PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_room_results(all_results):
    """4 panels per condition + summary overlay."""
    conditions = list(all_results.keys())
    n_cond = len(conditions)

    # Per-condition detail plot
    fig, axes = plt.subplots(3, n_cond, figsize=(5 * n_cond, 12))
    if n_cond == 1:
        axes = axes[:, None]

    for col, cond in enumerate(conditions):
        res = all_results[cond]
        n = res['n_depths']
        x = list(range(n))
        x_lbl = ['Emb'] + [f'L{i+1}' for i in range(n - 1)]

        ax = axes[0, col]
        ax.plot(x, res['freq']['cross_re'], 'o-', color='darkorange', lw=2, ms=8, label='cross_re')
        ax.plot(x, res['freq']['cross_im'], 's--', color='sienna', lw=1.5, ms=7, label='cross_im')
        ax.set_title(f'{cond}\ncross-power R²', fontsize=10, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(x_lbl)
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.text(0.5, 0.05, f'val MAE={res["val_mae"]:.2f}',
                transform=ax.transAxes, ha='center', fontsize=9, color='gray')

        ax = axes[1, col]
        ax.plot(x, res['freq']['phat_cos'], 'o-', color='royalblue', lw=2, ms=8, label='phat_cos')
        ax.plot(x, res['freq']['phat_sin'], 'v--', color='deepskyblue', lw=1.5, ms=7, label='phat_sin')
        ax.set_title(f'{cond}\nPHAT phase R²', fontsize=10, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(x_lbl)
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[2, col]
        ax.plot(x, res['global']['tau'], '^-', color='black', lw=2.5, ms=9, label='Global τ')
        ax.plot(x, res['freq']['cross_re'], 'o--', color='darkorange', lw=1.5, ms=7,
                alpha=0.7, label='per-freq cross_re')
        ax.set_title(f'{cond}\nτ decode + cross_re', fontsize=10, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(x_lbl)
        ax.set_ylim(-0.15, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for ax in axes.flat:
        ax.axhline(0, c='gray', lw=0.8, ls=':')
        ax.set_xlabel('Depth'); ax.set_ylabel('R²')

    plt.suptitle(
        'Room Validation — Reverberant Acoustic Environments (pyroomacoustics)\n'
        f'SNR={SNR_DB:.0f} dB, mic sep={MIC_SEP:.1f}m, fs={FS} Hz',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = 'results/room_validation_detail.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {path}")
    plt.close()

    # Summary: T60 vs key metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    t60_labels = list(all_results.keys())
    t60_centers = {'t60_0.2': 0.2, 't60_0.4': 0.4, 't60_0.6': 0.6, 'mixed': 0.42}
    x_pos = [t60_centers[c] for c in t60_labels]

    cross_d1 = [all_results[c]['freq']['cross_re'][1] for c in t60_labels]
    phat_max = [max(all_results[c]['freq']['phat_cos']) for c in t60_labels]
    tau_last = [all_results[c]['global']['tau'][-1] for c in t60_labels]
    val_maes = [all_results[c]['val_mae'] for c in t60_labels]

    ax = axes[0]
    ax.bar(range(len(t60_labels)), cross_d1, color='darkorange', alpha=0.8)
    ax.set_xticks(range(len(t60_labels))); ax.set_xticklabels(t60_labels, rotation=15)
    ax.set_ylabel('R²'); ax.set_ylim(0, 1.1)
    ax.set_title('cross_re @ D1\n(Phase 1: cross-power)', fontsize=10, fontweight='bold')
    ax.axhline(0.876, ls='--', color='red', lw=1.5, alpha=0.7, label='Synthetic baseline')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(range(len(t60_labels)), phat_max, color='royalblue', alpha=0.8)
    ax.set_xticks(range(len(t60_labels))); ax.set_xticklabels(t60_labels, rotation=15)
    ax.set_ylabel('R²'); ax.set_ylim(0, 1.1)
    ax.set_title('phat_cos (best depth)\n(Phase 2: PHAT whitening)', fontsize=10, fontweight='bold')
    ax.axhline(0.12, ls='--', color='red', lw=1.5, alpha=0.7, label='Synthetic baseline')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    ax.bar(range(len(t60_labels)), val_maes, color='gray', alpha=0.8)
    ax.set_xticks(range(len(t60_labels))); ax.set_xticklabels(t60_labels, rotation=15)
    ax.set_ylabel('Val MAE (samples)')
    ax.set_title('Task Performance\n(lower is better)', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Room Validation Summary — "Does the synthetic finding hold?"',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = 'results/room_validation_summary.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_results = {}

    for cond_name, t60_range in T60_CONDITIONS.items():
        print(f"\n{'━'*60}")
        print(f"  Condition: {cond_name}  (T60={t60_range[0]:.2f}–{t60_range[1]:.2f})")
        print(f"{'━'*60}")

        # Generate RIR pool
        print("  [1/4] Generating RIRs ...")
        t0 = time.time()
        rng_rir = np.random.RandomState(42)
        tr_pool = generate_rir_pool(500, t60_range, rng_rir)
        va_pool = generate_rir_pool(100, t60_range, np.random.RandomState(99))
        print(f"       {len(tr_pool)} train RIRs, {len(va_pool)} val RIRs "
              f"({time.time()-t0:.1f}s)")

        if len(tr_pool) < 50 or len(va_pool) < 10:
            print(f"  !! Not enough valid RIRs, skipping {cond_name}")
            continue

        # Build datasets
        print("  [2/4] Building datasets ...")
        t0 = time.time()
        tr_ds = RoomTDOADataset(tr_pool, N_TRAIN, SNR_DB, T, seed=42)
        va_ds = RoomTDOADataset(va_pool, N_VAL,   SNR_DB, T, seed=99)
        print(f"       train={len(tr_ds)}, val={len(va_ds)} ({time.time()-t0:.1f}s)")

        # Train
        print("  [3/4] Training Transformer ...")
        model = TransformerModel(F=F, d_model=64, n_layers=4, n_heads=4, d_ff=256)
        print(f"       params: {model.n_params:,}")
        t0 = time.time()
        hist = train_model(model, tr_ds, va_ds, epochs=EPOCHS, lr=3e-4, batch_size=1024)
        val_mae = hist['val_mae'][-1]
        print(f"       {time.time()-t0:.1f}s | val MAE: {val_mae:.3f}")
        torch.save(model.state_dict(), f'results/room_{cond_name}_model.pt')

        # Probe
        print("  [4/4] Probing ...")
        freq_p, global_p, n_depths = run_probes(model, va_ds, n_probe=min(3000, len(va_ds)))
        print(f"       cross_re: {[f'{v:.3f}' for v in freq_p['cross_re']]}")
        print(f"       phat_cos: {[f'{v:.3f}' for v in freq_p['phat_cos']]}")
        print(f"       tau:      {[f'{v:.3f}' for v in global_p['tau']]}")

        all_results[cond_name] = {
            'freq': freq_p, 'global': global_p,
            'n_depths': n_depths, 'val_mae': val_mae, 'history': hist,
        }

    # Save + plot
    torch.save(all_results, 'results/room_validation.pt')
    plot_room_results(all_results)

    # Summary table
    print(f"\n{'━'*60}")
    print("ROOM VALIDATION SUMMARY")
    print(f"{'━'*60}")
    print(f"{'Condition':12s}  {'val MAE':>8s}  {'cross_re@D1':>12s}  "
          f"{'phat_cos_max':>13s}  {'tau@last':>9s}")
    for cond, res in all_results.items():
        cr = res['freq']['cross_re'][1]
        pc = max(res['freq']['phat_cos'])
        tau = res['global']['tau'][-1]
        print(f"{cond:12s}  {res['val_mae']:>8.3f}  {cr:>12.3f}  {pc:>13.3f}  {tau:>9.3f}")

    print(f"\nFigures: results/room_validation_*.png")
    print(f"\n{'━'*60}")
    print("DONE")
    print(f"{'━'*60}")
