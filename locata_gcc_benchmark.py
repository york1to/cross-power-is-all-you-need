#!/usr/bin/env python3
"""
LOCATA GCC Benchmark
====================
Benchmark classical GCC methods (PHAT, Flat, Magnitude, Learned) on real
LOCATA recordings with GEOMETRIC ground-truth τ from OptiTrack positions.

This avoids circular reasoning: ground truth comes from mic/source geometry,
NOT from GCC-PHAT estimation.
"""

import os, sys, time
import numpy as np
import torch
from itertools import combinations
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

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

T          = 256
F          = T // 2 + 1   # 129
FS_TARGET  = 16000
TAU_MAX    = 30
C_SOUND    = 343.0        # speed of sound (m/s)
LOCATA_DIR = 'data/locata'
ARRAY_NAME = 'dicit'
N_MICS     = 15

os.makedirs('results', exist_ok=True)


# ─────────────────────────────────────────────────────────────
# GEOMETRIC GROUND TRUTH
# ─────────────────────────────────────────────────────────────

def load_positions(rec_dir, array_name):
    """Load mic and source positions from LOCATA position files.

    Returns:
        mic_pos: (N_MICS, 3) array of mic positions in meters
        src_pos: (3,) source position in meters
    """
    # Array positions
    arr_path = os.path.join(rec_dir, array_name,
                            f'position_array_{array_name}.txt')
    arr_data = np.loadtxt(arr_path, skiprows=1, delimiter='\t')
    # First row, columns: year,month,day,hour,minute,second,x,y,z,ref_vec*3,rot*9, then mic1_xyz...mic15_xyz
    # Mic positions start at column 21 (0-indexed): mic1_x, mic1_y, mic1_z, mic2_x, ...
    row = arr_data[0]  # use first timestamp (static array)
    mic_cols_start = 21
    mic_pos = row[mic_cols_start:mic_cols_start + N_MICS * 3].reshape(N_MICS, 3)

    # Source positions
    src_files = sorted([
        f for f in os.listdir(os.path.join(rec_dir, array_name))
        if f.startswith('position_source_')
    ])
    if not src_files:
        return None, None

    src_path = os.path.join(rec_dir, array_name, src_files[0])
    src_data = np.loadtxt(src_path, skiprows=1, delimiter='\t')
    # Columns: year,month,day,hour,minute,second,x,y,z,...
    src_pos = src_data[0, 6:9]  # x, y, z

    return mic_pos, src_pos


def compute_geometric_tdoa(mic_pos, src_pos, ch1, ch2, fs):
    """Compute geometric TDOA in samples from mic/source positions.

    τ = (d1 - d2) / c * fs
    where d1 = ||mic1 - src||, d2 = ||mic2 - src||
    """
    d1 = np.linalg.norm(mic_pos[ch1] - src_pos)
    d2 = np.linalg.norm(mic_pos[ch2] - src_pos)
    tau = (d1 - d2) / C_SOUND * fs
    return tau


# ─────────────────────────────────────────────────────────────
# GCC METHODS (from gcc_benchmark.py)
# ─────────────────────────────────────────────────────────────

def gcc_tdoa(mic1, mic2, tau_max, method='phat', learned_w=None):
    """Estimate TDOA using GCC with specified weighting."""
    X1 = np.fft.rfft(mic1)
    X2 = np.fft.rfft(mic2)
    G12 = X2 * np.conj(X1)
    eps = 1e-10

    if method == 'phat':
        W = 1.0 / (np.abs(G12) + eps)
    elif method == 'magnitude':
        W = np.abs(G12)
    elif method == 'flat':
        W = np.ones_like(G12, dtype=np.float64)
    elif method == 'learned':
        W = learned_w.astype(np.float64)
    else:
        raise ValueError(f"Unknown method: {method}")

    R = np.fft.irfft(W * G12, len(mic1))

    Tlen = len(mic1)
    candidates = np.concatenate([
        np.arange(0, tau_max + 1),
        np.arange(Tlen - tau_max, Tlen)
    ])
    best_idx = candidates[np.argmax(R[candidates])]
    tau_est = best_idx if best_idx <= tau_max else best_idx - Tlen
    return float(tau_est)


# ─────────────────────────────────────────────────────────────
# LOAD LEARNED PROFILE
# ─────────────────────────────────────────────────────────────

def load_learned_profile():
    """Load learned weighting profile from synthetic colored 0dB training."""
    path = 'results/gcc_benchmark.pt'
    if os.path.exists(path):
        d = torch.load(path, map_location='cpu', weights_only=False)
        key = 'colored_snr+0dB'
        if key in d and 'learned_profile' in d[key]:
            profile = np.array(d[key]['learned_profile'])
            print(f"  Loaded learned profile from {path} [{key}], shape={profile.shape}")
            return profile

    # Fallback: learned_weighting results
    path2 = 'results/learned_weighting_colored_snr+0dB.pt'
    if os.path.exists(path2):
        d = torch.load(path2, map_location='cpu', weights_only=False)
        if 'transformer' in d and 'learned_profile_avg' in d['transformer']:
            profile = np.array(d['transformer']['learned_profile_avg'])
            # Normalize to sum=1
            profile = profile / (profile.sum() + 1e-8)
            print(f"  Loaded learned profile from {path2}, shape={profile.shape}")
            return profile

    print("  WARNING: No learned profile found, skipping GCC-Learned")
    return None


# ─────────────────────────────────────────────────────────────
# MAIN BENCHMARK
# ─────────────────────────────────────────────────────────────

def run_locata_gcc_benchmark():
    """Run GCC benchmark on LOCATA with geometric ground truth."""
    task1_dir = os.path.join(LOCATA_DIR, 'dev', 'task1')
    if not os.path.isdir(task1_dir):
        raise FileNotFoundError(f"Not found: {task1_dir}")

    rec_dirs = sorted([
        os.path.join(task1_dir, d)
        for d in os.listdir(task1_dir)
        if os.path.isdir(os.path.join(task1_dir, d))
        and d.startswith('recording')
    ])

    learned_w = load_learned_profile()

    methods = ['phat', 'flat', 'magnitude']
    if learned_w is not None:
        methods.append('learned')

    all_errors = {m: [] for m in methods}
    all_taus_geo = []
    n_pairs_total = 0
    n_frames_total = 0

    from math import gcd

    for rec_dir in rec_dirs:
        rec_name = os.path.basename(rec_dir)
        audio_path = os.path.join(rec_dir, ARRAY_NAME,
                                  f'audio_array_{ARRAY_NAME}.wav')
        if not os.path.exists(audio_path):
            continue

        # Load positions
        mic_pos, src_pos = load_positions(rec_dir, ARRAY_NAME)
        if mic_pos is None or src_pos is None:
            print(f"  {rec_name}: no position files, skipping")
            continue

        # Load and resample audio
        audio, fs = sf.read(audio_path)
        if audio.ndim == 1 or audio.shape[1] < N_MICS:
            continue

        g = gcd(FS_TARGET, fs)
        up, down = FS_TARGET // g, fs // g
        audio_16k = resample_poly(audio, up, down, axis=0).astype(np.float64)
        n_frames = len(audio_16k) // T
        if n_frames < 10:
            continue

        n_valid = 0
        for ch1, ch2 in combinations(range(N_MICS), 2):
            tau_geo = compute_geometric_tdoa(mic_pos, src_pos, ch1, ch2,
                                             FS_TARGET)
            if abs(tau_geo) > TAU_MAX or abs(tau_geo) < 0.5:
                continue

            mic1 = audio_16k[:n_frames * T, ch1].reshape(n_frames, T)
            mic2 = audio_16k[:n_frames * T, ch2].reshape(n_frames, T)

            for i in range(n_frames):
                for m in methods:
                    tau_est = gcc_tdoa(
                        mic1[i], mic2[i], TAU_MAX, method=m,
                        learned_w=learned_w if m == 'learned' else None)
                    all_errors[m].append(abs(tau_est - tau_geo))
                all_taus_geo.append(tau_geo)

            n_valid += 1
            n_frames_total += n_frames

        n_pairs_total += n_valid
        print(f"  {rec_name}: {n_valid} valid pairs, "
              f"{n_valid * n_frames} frames")

    # Compute MAE
    results = {}
    print(f"\n{'='*50}")
    print(f"LOCATA GCC Benchmark Results")
    print(f"  {n_pairs_total} pairs, {n_frames_total} frames")
    print(f"  Ground truth: geometric (OptiTrack positions)")
    print(f"{'='*50}")
    for m in methods:
        mae = float(np.mean(all_errors[m]))
        median = float(np.median(all_errors[m]))
        results[m] = {
            'mae': mae,
            'median': median,
            'errors': np.array(all_errors[m]),
        }
        print(f"  {m:12s}: MAE = {mae:.3f}, median = {median:.3f}")

    # Also compare with GCC-PHAT ground truth for sanity check
    print(f"\n  Geometric τ range: [{min(all_taus_geo):.1f}, "
          f"{max(all_taus_geo):.1f}]")
    print(f"  Unique geometric τ: {len(set(round(t, 1) for t in all_taus_geo))}")

    return results, n_pairs_total, n_frames_total


if __name__ == '__main__':
    t_start = time.time()
    print("=" * 60)
    print("LOCATA GCC Benchmark (Geometric Ground Truth)")
    print("=" * 60)

    results, n_pairs, n_frames = run_locata_gcc_benchmark()

    # Save
    save_data = {
        'results': {m: {'mae': v['mae'], 'median': v['median']}
                    for m, v in results.items()},
        'n_pairs': n_pairs,
        'n_frames': n_frames,
        'ground_truth': 'geometric (OptiTrack)',
        'config': {
            'T': T, 'F': F, 'fs': FS_TARGET,
            'TAU_MAX': TAU_MAX, 'c_sound': C_SOUND,
            'array': ARRAY_NAME, 'n_mics': N_MICS,
        },
    }
    out_path = 'results/locata_gcc_benchmark.pt'
    torch.save(save_data, out_path)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")
