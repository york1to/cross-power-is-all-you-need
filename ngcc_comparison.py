#!/usr/bin/env python3
"""
NGCC-PHAT vs Neural GCC-Flat Comparison
=========================================
Compare neural networks that receive different GCC preprocessing as input:
  - NGCC-PHAT:    input = IFFT(X2·X1* / |X2·X1*|)  → MLP → τ
  - NGCC-Flat:    input = IFFT(X2·X1*)              → MLP → τ
  - NGCC-Mag:     input = IFFT(|X2·X1*| · X2·X1*)  → MLP → τ
  - NGCC-Learned: input = IFFT(W_learned · X2·X1*)  → MLP → τ

All use the same MLP architecture, only the GCC preprocessing differs.
This directly tests: does PHAT whitening help or hurt as neural network input?

Also includes:
  - Classical (argmax) baselines for each weighting
  - Per-frequency token Transformer (current approach) for reference
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────

T_SIG, TAU_MAX = 256, 30
F = T_SIG // 2 + 1
# Correlation function window: lags [-TAU_MAX, ..., +TAU_MAX]
CORR_LEN = 2 * TAU_MAX + 1  # 61


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


def compute_gcc_features(mic1, mic2, method='phat', learned_w=None):
    """
    Compute GCC cross-correlation and extract the [-TAU_MAX, +TAU_MAX] window.
    Returns: (CORR_LEN,) array = R[lag] for lag in [-TAU_MAX, ..., +TAU_MAX]
    """
    X1 = np.fft.rfft(mic1)
    X2 = np.fft.rfft(mic2)
    G12 = X2 * np.conj(X1)  # X2·X1* so IFFT peak at +τ
    eps = 1e-10

    if method == 'phat':
        W = 1.0 / (np.abs(G12) + eps)
    elif method == 'flat':
        W = np.ones_like(G12, dtype=np.float64)
    elif method == 'magnitude':
        W = np.abs(G12)
    elif method == 'learned':
        W = learned_w.astype(np.float64)
    else:
        raise ValueError(f"Unknown method: {method}")

    R = np.fft.irfft(W * G12, len(mic1))

    # Extract [-TAU_MAX, +TAU_MAX] window
    # Positive lags: R[0], R[1], ..., R[TAU_MAX]
    # Negative lags: R[T-TAU_MAX], ..., R[T-1]  → correspond to lag -TAU_MAX, ..., -1
    pos = R[:TAU_MAX + 1]             # lags 0..+TAU_MAX  (31,)
    neg = R[len(mic1) - TAU_MAX:]     # lags -TAU_MAX..-1 (30,)
    window = np.concatenate([neg, pos])  # lags -TAU_MAX..+TAU_MAX (61,)
    return window.astype(np.float32)


def to_tokens(mic1, mic2):
    """Per-frequency tokens for the Transformer baseline."""
    X1, X2 = np.fft.rfft(mic1), np.fft.rfft(mic2)
    t = np.stack([X1.real, X1.imag, X2.real, X2.imag], axis=1).astype(np.float32)
    return t / (np.std(t) + 1e-8)


# ─────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────

class NGCCDataset(Dataset):
    """Dataset that precomputes GCC features for multiple methods."""

    def __init__(self, N, tau_max=30, snr_db=0., T=256, noise_type='white',
                 seed=0, methods=None, learned_w=None):
        if methods is None:
            methods = ['phat', 'flat']
        rng = np.random.RandomState(seed)
        taus = rng.uniform(-tau_max, tau_max, N).astype(np.float32)

        # Precompute all GCC features
        gcc_feats = {m: [] for m in methods}
        tokens_list = []

        for tau in tqdm(taus, desc='  data', leave=False):
            mic1, mic2 = make_pair(tau, snr_db, T, noise_type, rng)
            for m in methods:
                gcc_feats[m].append(compute_gcc_features(
                    mic1, mic2, method=m, learned_w=learned_w))
            tokens_list.append(to_tokens(mic1, mic2))

        self.gcc = {m: torch.from_numpy(np.stack(v)) for m, v in gcc_feats.items()}
        self.tokens = torch.from_numpy(np.stack(tokens_list))
        self.taus_norm = torch.from_numpy(taus / tau_max)
        self.taus_raw = torch.from_numpy(taus)
        self.T, self.F, self.tau_max = T, T // 2 + 1, tau_max

    def __len__(self):
        return len(self.taus_norm)

    def __getitem__(self, i):
        return i  # index-based access, actual data retrieved by method


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

class NGCC_MLP(nn.Module):
    """MLP that takes GCC correlation window as input and predicts τ."""

    def __init__(self, input_dim=CORR_LEN, hidden=256, n_layers=3, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    """Per-frequency token Transformer (reference)."""

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
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        B = x.shape[0]
        h = self.input_proj(x) + self.freq_embed(torch.arange(self.F, device=x.device))
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        for layer in self.layers_:
            h = layer(h)
        return self.head(self.norm_out(h[:, 0]))


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_ngcc(model, train_feats, train_labels, val_feats, val_labels,
               tau_max, epochs=120, lr=3e-4, batch_size=1024):
    """Train NGCC model on precomputed GCC features."""
    model.to(device)
    N_train = len(train_feats)
    N_val = len(val_feats)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    total_steps  = epochs * ((N_train + batch_size - 1) // batch_size)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * p))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.HuberLoss()

    pbar = tqdm(range(epochs), desc='  train', leave=False)
    for _ in pbar:
        model.train()
        perm = torch.randperm(N_train)
        for s in range(0, N_train, batch_size):
            idx = perm[s:s + batch_size]
            x = train_feats[idx].to(device)
            y = train_labels[idx].unsqueeze(1).to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            va_pred = []
            for s in range(0, N_val, batch_size):
                x = val_feats[s:s + batch_size].to(device)
                va_pred.append(model(x).cpu())
            va_pred = torch.cat(va_pred).squeeze()
            va_mae = (va_pred - val_labels).abs().mean().item() * tau_max
        pbar.set_postfix({'va': f'{va_mae:.3f}'})

    return va_mae


def train_transformer(model, train_ds, val_ds, epochs=120, lr=3e-4, batch_size=1024):
    """Train Transformer on per-frequency tokens."""
    model.to(device)
    tr = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    va = DataLoader(val_ds, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    total_steps  = epochs * len(tr)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * p))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.HuberLoss()
    pbar = tqdm(range(epochs), desc='  train', leave=False)
    for _ in pbar:
        model.train()
        for toks, tau in tr:
            toks, tau = toks.to(device), tau.to(device)
            pred = model(toks)
            loss = loss_fn(pred, tau)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
        model.eval()
        va_mae = []
        with torch.no_grad():
            for toks, tau in va:
                toks, tau = toks.to(device), tau.to(device)
                va_mae.append((model(toks) - tau).abs().mean().item() * val_ds.tau_max)
        pbar.set_postfix({'va': f'{np.mean(va_mae):.3f}'})
    return np.mean(va_mae)


def extract_gradient_profile(model, dataset, n_samples=2000, batch_size=64):
    """Extract gradient-based frequency weighting profile from Transformer."""
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
    profile = grad_w.mean(axis=0)
    profile = profile / (profile.sum() + 1e-8)
    return profile


# ─────────────────────────────────────────────────────────────
# CLASSICAL GCC BASELINE (argmax)
# ─────────────────────────────────────────────────────────────

def classical_gcc_eval(n_test, snr_db, noise_type, learned_w=None, rng_seed=777):
    """Evaluate classical GCC-argmax for comparison."""
    rng = np.random.RandomState(rng_seed)
    methods = ['phat', 'flat', 'magnitude']
    if learned_w is not None:
        methods.append('learned')
    results = {m: [] for m in methods}

    for _ in range(n_test):
        tau_true = rng.uniform(-TAU_MAX, TAU_MAX)
        mic1, mic2 = make_pair(tau_true, snr_db, T_SIG, noise_type, rng)
        for m in methods:
            feat = compute_gcc_features(mic1, mic2, method=m,
                                        learned_w=learned_w if m == 'learned' else None)
            # argmax over the window: index TAU_MAX = lag 0
            tau_est = np.argmax(feat) - TAU_MAX
            results[m].append(abs(tau_est - tau_true))

    return {m: float(np.mean(v)) for m, v in results.items()}


# ─────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    N_TRAIN, N_VAL = 50_000, 5_000
    N_TEST_CLASSICAL = 10_000
    EPOCHS = 120
    SNR_LEVELS = [20, 10, 5, 0, -5, -10]
    NOISE_TYPES = ['white', 'colored']
    NGCC_METHODS = ['phat', 'flat', 'magnitude']  # learned added per-condition

    all_results = {}

    print(f"\n{'━'*70}")
    print("  NGCC-PHAT vs Neural GCC-Flat Comparison")
    print(f"{'━'*70}")

    for noise_type in NOISE_TYPES:
        for snr_db in SNR_LEVELS:
            tag = f"{noise_type}_snr{snr_db:+d}dB"
            print(f"\n{'─'*60}")
            print(f"  {tag}")
            print(f"{'─'*60}")

            # ── Step 1: Train per-frequency Transformer (reference) ──
            print("  [1] Training per-freq Transformer (reference) ...")

            class SimpleTokenDS(Dataset):
                def __init__(self, tokens, taus_norm, tau_max, T, F):
                    self.tokens = tokens
                    self.taus_norm = taus_norm
                    self.tau_max = tau_max
                    self.T, self.F = T, F
                def __len__(self):
                    return len(self.taus_norm)
                def __getitem__(self, i):
                    return self.tokens[i], self.taus_norm[i:i+1]

            # Generate raw data for all methods
            rng_tr = np.random.RandomState(42)
            rng_va = np.random.RandomState(99)
            taus_tr = rng_tr.uniform(-TAU_MAX, TAU_MAX, N_TRAIN).astype(np.float32)
            taus_va = rng_va.uniform(-TAU_MAX, TAU_MAX, N_VAL).astype(np.float32)

            print("    Generating training data ...")
            # Collect raw mic signals for GCC computation
            gcc_tr = {m: [] for m in NGCC_METHODS}
            tokens_tr = []
            for tau in tqdm(taus_tr, desc='    train data', leave=False):
                mic1, mic2 = make_pair(tau, float(snr_db), T_SIG, noise_type, rng_tr)
                tokens_tr.append(to_tokens(mic1, mic2))
                for m in NGCC_METHODS:
                    gcc_tr[m].append(compute_gcc_features(mic1, mic2, method=m))

            print("    Generating val data ...")
            gcc_va = {m: [] for m in NGCC_METHODS}
            tokens_va = []
            for tau in tqdm(taus_va, desc='    val data', leave=False):
                mic1, mic2 = make_pair(tau, float(snr_db), T_SIG, noise_type, rng_va)
                tokens_va.append(to_tokens(mic1, mic2))
                for m in NGCC_METHODS:
                    gcc_va[m].append(compute_gcc_features(mic1, mic2, method=m))

            # Convert to tensors
            tokens_tr_t = torch.from_numpy(np.stack(tokens_tr))
            tokens_va_t = torch.from_numpy(np.stack(tokens_va))
            taus_tr_norm = torch.from_numpy(taus_tr / TAU_MAX)
            taus_va_norm = torch.from_numpy(taus_va / TAU_MAX)
            taus_va_raw  = torch.from_numpy(taus_va)

            gcc_tr_t = {m: torch.from_numpy(np.stack(v)) for m, v in gcc_tr.items()}
            gcc_va_t = {m: torch.from_numpy(np.stack(v)) for m, v in gcc_va.items()}

            # Normalize GCC features (per-method, to have ~unit variance)
            gcc_tr_norm = {}
            gcc_va_norm = {}
            gcc_stats = {}
            for m in NGCC_METHODS:
                mu = gcc_tr_t[m].mean()
                std = gcc_tr_t[m].std() + 1e-8
                gcc_tr_norm[m] = (gcc_tr_t[m] - mu) / std
                gcc_va_norm[m] = (gcc_va_t[m] - mu) / std
                gcc_stats[m] = (mu, std)

            # Train Transformer
            tr_ds = SimpleTokenDS(tokens_tr_t, taus_tr_norm, TAU_MAX, T_SIG, F)
            va_ds = SimpleTokenDS(tokens_va_t, taus_va_norm, TAU_MAX, T_SIG, F)
            xfmr = TransformerModel(F=F)
            xfmr_mae = train_transformer(xfmr, tr_ds, va_ds, epochs=EPOCHS)
            print(f"       Transformer MAE: {xfmr_mae:.3f} ({xfmr.n_params:,} params)")

            # ── Step 2: Extract learned profile & compute learned GCC features ──
            print("  [2] Extracting learned weighting profile ...")
            learned_w = extract_gradient_profile(xfmr, va_ds, n_samples=2000)
            print(f"       Profile peak at bin {np.argmax(learned_w)}")

            # Compute learned GCC features for train/val
            print("    Computing GCC-Learned features ...")
            gcc_tr_learned = []
            rng_tr2 = np.random.RandomState(42)  # same seed to reproduce same data
            taus_tr2 = rng_tr2.uniform(-TAU_MAX, TAU_MAX, N_TRAIN).astype(np.float32)
            for tau in tqdm(taus_tr2, desc='    train learned', leave=False):
                mic1, mic2 = make_pair(tau, float(snr_db), T_SIG, noise_type, rng_tr2)
                gcc_tr_learned.append(compute_gcc_features(
                    mic1, mic2, method='learned', learned_w=learned_w))

            gcc_va_learned = []
            rng_va2 = np.random.RandomState(99)
            taus_va2 = rng_va2.uniform(-TAU_MAX, TAU_MAX, N_VAL).astype(np.float32)
            for tau in tqdm(taus_va2, desc='    val learned', leave=False):
                mic1, mic2 = make_pair(tau, float(snr_db), T_SIG, noise_type, rng_va2)
                gcc_va_learned.append(compute_gcc_features(
                    mic1, mic2, method='learned', learned_w=learned_w))

            gcc_tr_t['learned'] = torch.from_numpy(np.stack(gcc_tr_learned))
            gcc_va_t['learned'] = torch.from_numpy(np.stack(gcc_va_learned))
            mu_l = gcc_tr_t['learned'].mean()
            std_l = gcc_tr_t['learned'].std() + 1e-8
            gcc_tr_norm['learned'] = (gcc_tr_t['learned'] - mu_l) / std_l
            gcc_va_norm['learned'] = (gcc_va_t['learned'] - mu_l) / std_l

            methods_all = NGCC_METHODS + ['learned']

            # ── Step 3: Train NGCC models for each method ──
            condition_results = {
                'transformer_mae': xfmr_mae,
                'learned_profile': learned_w,
            }

            for m in methods_all:
                print(f"  [3] Training NGCC-{m.upper()} ...")
                model = NGCC_MLP(input_dim=CORR_LEN, hidden=256, n_layers=3)
                mae = train_ngcc(
                    model, gcc_tr_norm[m], taus_tr_norm,
                    gcc_va_norm[m], taus_va_norm,
                    tau_max=TAU_MAX, epochs=EPOCHS, batch_size=1024)
                condition_results[f'ngcc_{m}'] = mae
                if m == methods_all[0]:
                    print(f"       NGCC-{m.upper()} MAE: {mae:.3f} ({model.n_params:,} params)")
                else:
                    print(f"       NGCC-{m.upper()} MAE: {mae:.3f}")

            # ── Step 4: Classical argmax baselines ──
            print("  [4] Classical GCC (argmax) ...")
            classical = classical_gcc_eval(
                N_TEST_CLASSICAL, float(snr_db), noise_type, learned_w, rng_seed=777)
            condition_results['classical'] = classical

            all_results[tag] = condition_results

            # ── Print comparison table ──
            print(f"\n       {'Method':25s}  {'MAE':>8s}")
            print(f"       {'─'*40}")
            # Classical
            for m in ['phat', 'flat', 'magnitude', 'learned']:
                if m in classical:
                    print(f"       {'Classical '+m.upper():25s}  {classical[m]:>8.3f}")
            print(f"       {'─'*40}")
            # Neural GCC
            for m in methods_all:
                print(f"       {'NGCC-'+m.upper():25s}  {condition_results[f'ngcc_{m}']:>8.3f}")
            print(f"       {'─'*40}")
            print(f"       {'Transformer (per-freq)':25s}  {xfmr_mae:>8.3f}")

    # ── Save ──
    torch.save(all_results, 'results/ngcc_comparison.pt')

    # ── Summary Table ──
    print(f"\n{'━'*70}")
    print("NGCC COMPARISON SUMMARY")
    print(f"{'━'*70}")

    header = (f"  {'Condition':20s}  {'Cls-PHAT':>9s}  {'Cls-Flat':>9s}"
              f"  {'NGCC-PHAT':>10s}  {'NGCC-Flat':>10s}  {'NGCC-Mag':>9s}"
              f"  {'NGCC-Learn':>10s}  {'Transformer':>11s}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for noise_type in NOISE_TYPES:
        for snr_db in SNR_LEVELS:
            tag = f"{noise_type}_snr{snr_db:+d}dB"
            r = all_results[tag]
            c = r['classical']
            print(f"  {tag:20s}"
                  f"  {c['phat']:>9.3f}  {c['flat']:>9.3f}"
                  f"  {r['ngcc_phat']:>10.3f}  {r['ngcc_flat']:>10.3f}"
                  f"  {r['ngcc_magnitude']:>9.3f}"
                  f"  {r['ngcc_learned']:>10.3f}  {r['transformer_mae']:>11.3f}")

    # ── Key comparison: NGCC-PHAT vs NGCC-Flat ──
    print(f"\n  NGCC-PHAT vs NGCC-Flat improvement:")
    print(f"  {'Condition':20s}  {'NGCC-PHAT':>10s}  {'NGCC-Flat':>10s}  {'Flat wins by':>12s}")
    for noise_type in NOISE_TYPES:
        for snr_db in SNR_LEVELS:
            tag = f"{noise_type}_snr{snr_db:+d}dB"
            r = all_results[tag]
            phat_mae = r['ngcc_phat']
            flat_mae = r['ngcc_flat']
            if phat_mae > 0:
                delta = (phat_mae - flat_mae) / phat_mae * 100
            else:
                delta = 0
            sign = '+' if delta > 0 else ''
            print(f"  {tag:20s}  {phat_mae:>10.3f}  {flat_mae:>10.3f}  {sign}{delta:>10.1f}%")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {
        'ngcc_phat': 'royalblue', 'ngcc_flat': 'gray',
        'ngcc_magnitude': 'darkorange', 'ngcc_learned': 'crimson',
        'transformer': 'black',
        'cls_phat': 'royalblue', 'cls_flat': 'gray',
    }
    labels = {
        'ngcc_phat': 'NGCC-PHAT', 'ngcc_flat': 'NGCC-Flat',
        'ngcc_magnitude': 'NGCC-Mag', 'ngcc_learned': 'NGCC-Learned',
        'transformer': 'Transformer (per-freq)',
        'cls_phat': 'Classical PHAT', 'cls_flat': 'Classical Flat',
    }

    for col, noise_type in enumerate(NOISE_TYPES):
        ax = axes[col]

        # Neural GCC methods
        for key in ['ngcc_phat', 'ngcc_flat', 'ngcc_magnitude', 'ngcc_learned']:
            maes = [all_results[f'{noise_type}_snr{s:+d}dB'][key] for s in SNR_LEVELS]
            ls = '-' if 'ngcc' in key else '--'
            ax.plot(SNR_LEVELS, maes, 'o' + ls, color=colors[key],
                    lw=2.5, ms=8, label=labels[key], zorder=5 if 'learned' in key else 3)

        # Transformer reference
        xfmr_maes = [all_results[f'{noise_type}_snr{s:+d}dB']['transformer_mae']
                      for s in SNR_LEVELS]
        ax.plot(SNR_LEVELS, xfmr_maes, 's--', color='black', lw=2, ms=7,
                label='Transformer (per-freq)', alpha=0.7)

        # Classical baselines (dashed, thin)
        for cm in ['phat', 'flat']:
            cls_maes = [all_results[f'{noise_type}_snr{s:+d}dB']['classical'][cm]
                        for s in SNR_LEVELS]
            ax.plot(SNR_LEVELS, cls_maes, 'x:', color=colors[f'cls_{cm}'],
                    lw=1.5, ms=6, label=f'Classical {cm.upper()}', alpha=0.5)

        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel('MAE (samples)', fontsize=11)
        ax.set_title(f'{noise_type.title()} Noise', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle(
        'NGCC-PHAT vs Neural GCC-Flat — Does PHAT Preprocessing Help Neural TDOA?\n'
        'Same MLP architecture, different GCC weighting as input',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/ngcc_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n  → Saved: results/ngcc_comparison.png")
    plt.close()

    print(f"\n{'━'*70}")
    print("DONE")
    print(f"{'━'*70}")
