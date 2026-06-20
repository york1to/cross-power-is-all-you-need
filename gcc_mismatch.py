#!/usr/bin/env python3
"""
GCC-Learned SNR Mismatch Analysis
====================================
Key question: Is the learned weighting profile universal or SNR-specific?

Method: Take each learned profile (extracted at SNR_train) and evaluate it
as a GCC weighting at ALL other SNR conditions. This gives a full
(SNR_train × SNR_test) cross-evaluation matrix.

Also evaluate a single "universal" profile (from colored 0dB) across all conditions.

This script loads profiles from gcc_benchmark.pt — run AFTER gcc_benchmark.py completes.
No training needed, just GCC evaluation (~5 min total).
"""

import os
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

os.makedirs('results', exist_ok=True)

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


def gcc_tdoa(mic1, mic2, tau_max, method='phat', learned_w=None):
    X1 = np.fft.rfft(mic1)
    X2 = np.fft.rfft(mic2)
    G12 = X2 * np.conj(X1)  # X2·X1* so peak is at +τ
    eps = 1e-10

    if method == 'phat':
        W = 1.0 / (np.abs(G12) + eps)
    elif method == 'flat':
        W = np.ones_like(G12, dtype=np.float64)
    elif method == 'learned':
        W = learned_w.astype(np.float64)
    else:
        raise ValueError(f"Unknown method: {method}")

    R = np.fft.irfft(W * G12, len(mic1))
    T_len = len(mic1)
    candidates = np.concatenate([
        np.arange(0, tau_max + 1),
        np.arange(T_len - tau_max, T_len)
    ])
    best_idx = candidates[np.argmax(R[candidates])]
    tau_est = best_idx if best_idx <= tau_max else best_idx - T_len
    return float(tau_est)


def evaluate_gcc(n_test, snr_db, noise_type, method, learned_w=None, rng_seed=777):
    rng = np.random.RandomState(rng_seed)
    errors = []
    for _ in range(n_test):
        tau_true = rng.uniform(-TAU_MAX, TAU_MAX)
        mic1, mic2 = make_pair(tau_true, snr_db, T, noise_type, rng)
        tau_est = gcc_tdoa(mic1, mic2, TAU_MAX, method=method, learned_w=learned_w)
        errors.append(abs(tau_est - tau_true))
    return float(np.mean(errors))


if __name__ == '__main__':

    # Load profiles from main benchmark
    data_path = 'results/gcc_benchmark.pt'
    print(f"Loading profiles from {data_path} ...")
    all_results = torch.load(data_path, map_location='cpu')

    SNR_LEVELS = [20, 10, 5, 0, -5, -10]
    NOISE_TYPES = ['white', 'colored']
    N_TEST = 10_000

    # ── Part 1: Cross-SNR matrix for colored noise ──
    print(f"\n{'━'*70}")
    print("  PART 1: Cross-SNR Mismatch Matrix (colored noise)")
    print(f"{'━'*70}")

    # Extract profiles
    profiles = {}
    for snr in SNR_LEVELS:
        tag = f"colored_snr{snr:+d}dB"
        if tag in all_results and 'learned_profile' in all_results[tag]:
            profiles[snr] = all_results[tag]['learned_profile']

    if not profiles:
        print("  ERROR: No profiles found in gcc_benchmark.pt")
        exit(1)

    # Cross-evaluation matrix: rows = profile_snr, cols = test_snr
    matrix = np.full((len(SNR_LEVELS), len(SNR_LEVELS)), np.nan)
    phat_baseline = np.full(len(SNR_LEVELS), np.nan)
    flat_baseline = np.full(len(SNR_LEVELS), np.nan)

    for j, test_snr in enumerate(SNR_LEVELS):
        print(f"\n  Testing at SNR={test_snr:+d}dB ...")

        # Baselines (only need once per test_snr)
        phat_baseline[j] = evaluate_gcc(N_TEST, float(test_snr), 'colored', 'phat')
        flat_baseline[j]  = evaluate_gcc(N_TEST, float(test_snr), 'colored', 'flat')
        print(f"    PHAT={phat_baseline[j]:.3f}  Flat={flat_baseline[j]:.3f}")

        for i, train_snr in enumerate(SNR_LEVELS):
            if train_snr in profiles:
                mae = evaluate_gcc(N_TEST, float(test_snr), 'colored', 'learned',
                                   learned_w=profiles[train_snr])
                matrix[i, j] = mae
                marker = " ◀ matched" if train_snr == test_snr else ""
                print(f"    Profile@{train_snr:+d}dB → {mae:.3f}{marker}")

    # ── Part 2: Universal profile (colored 0dB) across all conditions ──
    print(f"\n{'━'*70}")
    print("  PART 2: Universal Profile (colored 0dB) Across All Conditions")
    print(f"{'━'*70}")

    universal_profile = profiles.get(0)
    if universal_profile is None:
        print("  ERROR: No 0dB profile found")
        exit(1)

    universal_results = {}
    for noise_type in NOISE_TYPES:
        for snr in SNR_LEVELS:
            tag = f"{noise_type}_snr{snr:+d}dB"
            learned_mae = evaluate_gcc(N_TEST, float(snr), noise_type, 'learned',
                                       learned_w=universal_profile)
            phat_mae = evaluate_gcc(N_TEST, float(snr), noise_type, 'phat')
            flat_mae = evaluate_gcc(N_TEST, float(snr), noise_type, 'flat')
            universal_results[tag] = {
                'learned_universal': learned_mae,
                'phat': phat_mae,
                'flat': flat_mae,
            }
            print(f"  {tag:25s}  PHAT={phat_mae:.3f}  Flat={flat_mae:.3f}  "
                  f"Learned(0dB)={learned_mae:.3f}")

    # ── Save ──
    mismatch_data = {
        'cross_matrix': matrix,
        'snr_levels': SNR_LEVELS,
        'phat_baseline': phat_baseline,
        'flat_baseline': flat_baseline,
        'universal_results': universal_results,
        'profiles': profiles,
    }
    torch.save(mismatch_data, 'results/gcc_mismatch.pt')

    # ── Plot 1: Cross-SNR heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap of absolute MAE
    ax = axes[0]
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=min(20, np.nanmax(matrix)))
    ax.set_xticks(range(len(SNR_LEVELS)))
    ax.set_xticklabels([f'{s:+d}' for s in SNR_LEVELS])
    ax.set_yticks(range(len(SNR_LEVELS)))
    ax.set_yticklabels([f'{s:+d}' for s in SNR_LEVELS])
    ax.set_xlabel('Test SNR (dB)', fontsize=11)
    ax.set_ylabel('Profile extracted at SNR (dB)', fontsize=11)
    ax.set_title('GCC-Learned MAE (colored noise)\nRow = training SNR, Col = test SNR',
                 fontsize=11, fontweight='bold')
    # Annotate cells
    for i in range(len(SNR_LEVELS)):
        for j in range(len(SNR_LEVELS)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = 'white' if v > 8 else 'black'
                weight = 'bold' if i == j else 'normal'
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight=weight)
    # Mark diagonal
    for k in range(len(SNR_LEVELS)):
        ax.add_patch(plt.Rectangle((k-0.5, k-0.5), 1, 1, fill=False,
                                    edgecolor='blue', linewidth=2))
    plt.colorbar(im, ax=ax, label='MAE (samples)')

    # Comparison: matched vs universal vs baselines
    ax = axes[1]
    matched_mae = [matrix[i, i] for i in range(len(SNR_LEVELS))]
    universal_colored = [universal_results[f'colored_snr{s:+d}dB']['learned_universal']
                         for s in SNR_LEVELS]

    ax.plot(SNR_LEVELS, phat_baseline, 'o-', color='royalblue', lw=2, ms=7, label='GCC-PHAT')
    ax.plot(SNR_LEVELS, flat_baseline, 's-', color='gray', lw=2, ms=7, label='GCC-Flat')
    ax.plot(SNR_LEVELS, matched_mae, 'D-', color='crimson', lw=2.5, ms=8,
            label='GCC-Learned (matched SNR)')
    ax.plot(SNR_LEVELS, universal_colored, '^--', color='darkorange', lw=2, ms=8,
            label='GCC-Learned (universal, 0dB)')

    ax.set_xlabel('Test SNR (dB)', fontsize=11)
    ax.set_ylabel('MAE (samples)', fontsize=11)
    ax.set_title('Matched vs Universal Learned Profile\n(colored noise)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.suptitle('GCC-Learned: SNR Mismatch Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/gcc_mismatch.png', dpi=150, bbox_inches='tight')
    print(f"\n  → Saved: results/gcc_mismatch.png")
    plt.close()

    # ── Summary ──
    print(f"\n{'━'*70}")
    print("MISMATCH ANALYSIS SUMMARY")
    print(f"{'━'*70}")

    print(f"\n  Cross-SNR matrix (colored noise, MAE in samples):")
    col_header = "Profile\\Test"
    header = f"  {col_header:>12s}" + "".join(f"  {s:>+5d}dB" for s in SNR_LEVELS)
    print(header)
    for i, train_snr in enumerate(SNR_LEVELS):
        row = f"  {train_snr:>+9d}dB  "
        for j in range(len(SNR_LEVELS)):
            v = matrix[i, j]
            marker = '*' if i == j else ' '
            row += f"  {v:>6.2f}{marker}"
        print(row)
    print(f"  {'PHAT':>12s}" + "".join(f"  {v:>7.2f}" for v in phat_baseline))
    print(f"  {'Flat':>12s}" + "".join(f"  {v:>7.2f}" for v in flat_baseline))

    print(f"\n  Universal profile (colored 0dB) applied across all conditions:")
    print(f"  {'Condition':25s}  {'PHAT':>8s}  {'Flat':>8s}  {'Learned':>8s}  {'Best':>8s}")
    for noise_type in NOISE_TYPES:
        for snr in SNR_LEVELS:
            tag = f"{noise_type}_snr{snr:+d}dB"
            r = universal_results[tag]
            best = min(r, key=r.get)
            print(f"  {tag:25s}  {r['phat']:>8.3f}  {r['flat']:>8.3f}  "
                  f"{r['learned_universal']:>8.3f}  {best:>8s}")

    print(f"\n{'━'*70}")
    print("DONE")
    print(f"{'━'*70}")
