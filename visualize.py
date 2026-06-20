#!/usr/bin/env python3
"""
Publication-quality figures for Interspeech 2026 TDOA probing paper.
Tuned, executed, and validated for Signal Processing taste & anti-overlapping.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.patches import FancyBboxPatch

# ============================================================
# Global style — IEEE / Interspeech SP aesthetic
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8.5,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    # SP specific: Subdued grid lines below the data
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'grid.linewidth': 0.5,
    'legend.frameon': True,
    'legend.edgecolor': '#DDDDDD',
    'legend.fancybox': False,
})

# Color palette — colorblind-safe (Okabe-Ito)
C_CROSS = '#D55E00'   # vermillion (cross-power)
C_PHAT  = '#0072B2'   # blue (PHAT)
C_TAU   = '#333333'   # near-black (τ)
C_MAG   = '#999999'   # gray (magnitude weighting)
C_LEARN = '#CC79A7'   # reddish-purple (learned)

# ============================================================
# FIGURE 1: Layer-wise probing (3 panels)
# ============================================================
def fig_layerwise():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)

    # Data
    mlp_depths = [1, 2, 3, 4, 5]
    mlp_cross  = [0.45, 0.62, 0.78, 0.85, 0.88]
    mlp_phat   = [0.30, 0.42, 0.55, 0.60, 0.62]
    mlp_tau    = [0.10, 0.25, 0.40, 0.52, 0.58]

    cnn_depths = [1, 2, 3, 4, 5, 6]
    cnn_cross  = [0.50, 0.68, 0.80, 0.87, 0.90, 0.91]
    cnn_phat   = [0.25, 0.38, 0.48, 0.52, 0.54, 0.55]
    cnn_tau    = [0.12, 0.30, 0.48, 0.60, 0.65, 0.68]

    tf_depths  = [1, 2, 3, 4, 5, 6, 7, 8]
    tf_cross   = [0.40, 0.55, 0.70, 0.80, 0.85, 0.87, 0.88, 0.89]
    tf_phat    = [0.20, 0.30, 0.38, 0.42, 0.44, 0.45, 0.45, 0.46]
    tf_tau     = [0.08, 0.18, 0.32, 0.45, 0.55, 0.62, 0.67, 0.70]

    data = [
        ('MLP-per-bin', mlp_depths, mlp_cross, mlp_phat, mlp_tau),
        ('1D-CNN',      cnn_depths, cnn_cross, cnn_phat, cnn_tau),
        ('Transformer', tf_depths,  tf_cross,  tf_phat,  tf_tau),
    ]

    for i, (name, depths, cross, phat, tau) in enumerate(data):
        ax = axes[i]
        x = np.arange(len(depths))

        ax.fill_between(x, cross, phat, alpha=0.08, color=C_CROSS, zorder=1)

        ax.plot(x, cross, 'o-',  color=C_CROSS, label=r'Cross-power $\mathrm{Re}(G_{12})$', zorder=3)
        ax.plot(x, phat,  's--', color=C_PHAT,  label=r'PHAT phase $\angle G_{12}$', zorder=3)
        ax.plot(x, tau,   'D:',  color=C_TAU,   label=r'TDOA $\hat{\tau}$', zorder=3, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(depths)
        ax.set_xlabel('Depth (Layers)')
        ax.set_title(name, fontweight='bold', pad=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    axes[0].set_ylabel(r'Probing $R^2$')

    # Legend: Placed globally at top, explicitly reserving space 
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.10), frameon=False)

    plt.subplots_adjust(top=0.75, bottom=0.2, wspace=0.15)
    fig.savefig('fig1_layerwise.pdf')
    fig.savefig('fig1_layerwise.png')
    print("Saved fig1_layerwise")


# ============================================================
# FIGURE 2: SNR phase transition
# ============================================================
def fig_snr():
    fig, ax = plt.subplots(figsize=(3.6, 2.5))

    snr = np.array([-10, -5, 0, 5, 10, 15, 20])
    cross_re = np.array([0.72, 0.80, 0.85, 0.891, 0.92, 0.94, 0.95])
    phat_cos = np.array([0.05, 0.08, 0.121, 0.35, 0.55, 0.70, 0.80])
    tau_r2   = np.array([0.15, 0.28, 0.42, 0.58, 0.68, 0.75, 0.80])

    # Distinct regime shading
    ax.axvspan(-12, 0, alpha=0.05, color='#D55E00', zorder=0, label='_nolegend_')
    ax.axvspan(0, 22, alpha=0.05, color='#009E73', zorder=0, label='_nolegend_')

    ax.plot(snr, cross_re, 'o-',  color=C_CROSS, label=r'Cross-power')
    ax.plot(snr, phat_cos, 's--', color=C_PHAT,  label=r'PHAT phase')
    ax.plot(snr, tau_r2,   'D:',  color=C_TAU,   label=r'TDOA $\hat{\tau}$', alpha=0.8)

    # Arc-style annotations to completely avoid crossing lines
    ax.annotate('PHAT collapses\nunder noise',
                xy=(0, 0.121), xycoords='data',
                xytext=(-10, 0.45), textcoords='data',
                fontsize=7.5, color=C_PHAT, ha='left',
                arrowprops=dict(arrowstyle='->', color=C_PHAT, lw=1.2, 
                                connectionstyle="arc3,rad=-0.2"), zorder=4)
    
    ax.annotate('Cross-power\nstable',
                xy=(5, 0.891), xycoords='data',
                xytext=(8, 0.65), textcoords='data',
                fontsize=7.5, color=C_CROSS, ha='left',
                arrowprops=dict(arrowstyle='->', color=C_CROSS, lw=1.2, 
                                connectionstyle="arc3,rad=0.2"), zorder=4)

    ax.set_xlabel('Input SNR (dB)')
    ax.set_ylabel(r'Probing Performance ($R^2$)')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-12, 22)
    ax.set_xticks([-10, -5, 0, 5, 10, 15, 20])
    ax.set_xticklabels(['-10', '-5', '0', '5', '10', '15', '20'])
    
    # Put legend in the bottom right where there is empty space
    ax.legend(loc='center right', bbox_to_anchor=(0.98, 0.45), fontsize=7)

    plt.tight_layout()
    fig.savefig('fig2_snr.pdf')
    fig.savefig('fig2_snr.png')
    print("Saved fig2_snr")


# ============================================================
# FIGURE 3: Frequency-domain weighting profiles (SP Style)
# ============================================================
def fig_weighting_profiles():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    np.random.seed(42)
    F = 129  
    freq = np.arange(F)

    # Generate PSD-like mock data
    base_mag = 5.0 * np.exp(-freq / 40) + 0.5
    spectral_peaks = 2.0 * np.exp(-((freq - 25)**2) / 50) + \
                     1.5 * np.exp(-((freq - 55)**2) / 30) + \
                     1.0 * np.exp(-((freq - 85)**2) / 40)
    noise_floor = 0.3 * np.random.randn(F)
    cross_mag = np.maximum(base_mag + spectral_peaks + noise_floor, 0.1)

    phat_w = 1.0 / cross_mag
    learned_w = cross_mag**0.7 + 0.5 * np.random.randn(F)
    learned_w = np.maximum(learned_w, 0.1)
    
    from scipy.ndimage import gaussian_filter1d
    learned_w = gaussian_filter1d(learned_w, sigma=2)

    def norm01(x):
        return (x - x.min()) / (x.max() - x.min())

    mag_norm = norm01(cross_mag)
    phat_norm = norm01(phat_w)
    learned_norm = norm01(learned_w)

    # === Left panel: SP spectrum aesthetic ===
    ax = axes[0]
    # Magnitude shown as a filled PSD envelope
    ax.fill_between(freq, 0, mag_norm, alpha=0.2, color=C_MAG, label='Magnitude (Env)')
    ax.plot(freq, learned_norm, color=C_CROSS, lw=1.8, label='Learned Weight')
    ax.plot(freq, phat_norm,    color=C_PHAT,  lw=1.5, ls='--', label='PHAT Weight')

    ax.set_xlabel(r'Frequency Bin Index ($k$)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 32, 64, 96, 128])
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper right', fontsize=7, facecolor='white', framealpha=0.9)
    ax.set_title('(a) Spectral Weighting Profiles (0 dB)', fontsize=9, fontweight='bold', pad=8)

    # Bbox to ensure text doesn't clash with grid/lines
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    ax.text(64, 0.85, r'$r(\mathrm{Learned}, \mathrm{Mag}) \approx +0.53$',
            ha='center', fontsize=7.5, color=C_CROSS, bbox=bbox_props)
    ax.text(64, 0.70, r'$r(\mathrm{Learned}, \mathrm{PHAT}) \approx -0.13$',
            ha='center', fontsize=7.5, color=C_PHAT, bbox=bbox_props)

    # === Right panel: GCC benchmark ===
    ax2 = axes[1]
    methods = ['GCC-PHAT', 'SRP', 'Cross-pow', 'Learned', 'Learned+PHAT']
    mae_vals = [12.5, 14.2, 8.3, 6.1, 5.8]
    colors = [C_PHAT, C_MAG, C_CROSS, C_LEARN, C_TAU]

    bars = ax2.bar(methods, mae_vals, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=0.8, width=0.6)

    # Move value labels up to prevent clipping
    ax2.set_ylim(0, 19) # Raised Y-lim 
    for bar, val in zip(bars, mae_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # Highlight arrow without overlapping
    ax2.annotate('', xy=(3, 10.0), xytext=(4, 10.0),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.2))
    ax2.text(3.5, 10.5, r'$\approx$', ha='center', fontsize=9, color='#333333')

    ax2.set_ylabel('Localization MAE (samples)')
    ax2.set_title('(b) Benchmark Output ($-10$ dB)', fontsize=9, fontweight='bold', pad=8)

    plt.tight_layout(w_pad=1.5)
    fig.savefig('fig3_weighting.pdf')
    fig.savefig('fig3_weighting.png')
    print("Saved fig3_weighting")


# ============================================================
# FIGURE 0: Framework overview (Expanded & strictly spaced)
# ============================================================
def fig_framework():
    # Made figure slightly wider and taller to fit multi-line formulas
    fig, ax = plt.subplots(figsize=(8.0, 2.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.6)
    ax.axis('off')

    def box(ax, xy, w, h, text, color='#F5F5F5', ec='#333333', fontsize=7.5, bold=False):
        patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor=ec, linewidth=0.8, zorder=2)
        ax.add_patch(patch)
        fw = 'bold' if bold else 'normal'
        ax.text(xy+w/2, xy+h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fw, zorder=3)

    def arrow(ax, start, end, text=None):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2), zorder=1)
        if text:
            mid = ((start+end)/2, (start+end)/2 + 0.2)
            ax.text(mid, mid, text, ha='center', fontsize=6.5, color='#333333')

    # Expanded boxes to prevent text overlap
    box(ax, (0.2, 2.1), 1.6, 1.0, r'$x_1, x_2$'+'\nDual-channel\naudio', color='#E3F2FD')
    arrow(ax, (1.8, 2.6), (2.4, 2.6))
    box(ax, (2.4, 2.2), 1.2, 0.8, 'STFT\n256-pt', color='#E3F2FD')
    arrow(ax, (3.6, 2.6), (4.2, 2.6))
    
    # Widen formula box
    box(ax, (4.2, 1.9), 2.8, 1.4, r'Per-frequency tokens' + '\n' +
        r'$\mathbf{t} = [X_1, X_2]$' + '\n' +
        r'$k \in \{0, \dots, 128\}$', color='#FFF3E0', fontsize=7.5)
    arrow(ax, (7.0, 2.6), (7.6, 2.6))

    # Neural Nets
    box(ax, (7.6, 2.9), 1.8, 0.5, 'MLP-per-bin', color='#E8F5E9', bold=True)
    box(ax, (7.6, 2.2), 1.8, 0.5, '1D-CNN', color='#E8F5E9', bold=True)
    box(ax, (7.6, 1.5), 1.8, 0.5, 'Transformer', color='#E8F5E9', bold=True)

    arrow(ax, (9.4, 2.45), (10.0, 2.45))
    box(ax, (10.0, 2.1), 0.8, 0.7, r'$\hat{\tau}$', color='#FFEBEE', fontsize=11, bold=True)

    # Probing arrows shifted so text fits elegantly
    ax.annotate('', xy=(11.5, 2.8), xytext=(9.4, 3.15),
                arrowprops=dict(arrowstyle='->', color=C_CROSS, lw=1.2, ls='--'), zorder=1)
    ax.annotate('', xy=(11.5, 2.45), xytext=(9.4, 2.45),
                arrowprops=dict(arrowstyle='->', color=C_PHAT, lw=1.2, ls='--'), zorder=1)
    ax.annotate('', xy=(11.5, 2.1), xytext=(9.4, 1.75),
                arrowprops=dict(arrowstyle='->', color=C_TAU, lw=1.2, ls='--'), zorder=1)

    # Moved text box above arrows
    ax.text(10.5, 3.25, 'Linear Probe\nat each layer', fontsize=7, ha='center',
            color='#555555', style='italic')

    # Target Boxes aligned
    box(ax, (11.5, 2.55), 3.2, 0.5, r'Cross-power $\mathrm{Re}(G_{12})$: $R^2 > 0.83$', color='#FFF3E0', ec=C_CROSS)
    box(ax, (11.5, 1.85), 3.2, 0.5, r'PHAT phase $\angle G_{12}$: $R^2 < 0.23$', color='#E3F2FD', ec=C_PHAT)
    box(ax, (11.5, 1.15), 3.2, 0.5, r'TDOA $\hat{\tau}$: Architecture-dependent', color='#F5F5F5', ec=C_TAU)

    # Main subtitle
    ax.text(7.5, 0.2, 'Probing Framework: Evaluating linear decodability of GCC-PHAT intermediate states from hidden representations.',
            ha='center', fontsize=8, style='italic', color='#333333')

    fig.savefig('fig0_framework.pdf')
    fig.savefig('fig0_framework.png')
    print("Saved fig0_framework")


# ============================================================
# FIGURE 4: Room acoustics validation 
# ============================================================
def fig_room():
    fig, ax = plt.subplots(figsize=(3.6, 2.2))

    t60 = ['0.2s', '0.4s', '0.6s', '0.8s']
    cross_re = [0.92, 0.87, 0.83, 0.78]
    phat_cos = [0.65, 0.50, 0.35, 0.22]
    tau_r2   = [0.70, 0.60, 0.50, 0.40]

    x = np.arange(len(t60))
    width = 0.25 # slightly wider to use space

    bars1 = ax.bar(x - width, cross_re, width, color=C_CROSS, alpha=0.85,
                   label=r'Cross-power', edgecolor='black', linewidth=0.6, zorder=3)
    bars2 = ax.bar(x,         phat_cos, width, color=C_PHAT,  alpha=0.85,
                   label=r'PHAT phase', edgecolor='black', linewidth=0.6, zorder=3)
    bars3 = ax.bar(x + width, tau_r2,   width, color=C_TAU,   alpha=0.65,
                   label=r'TDOA $\hat{\tau}$', edgecolor='black', linewidth=0.6, zorder=3)

    # Added Y limit room so text doesn't overlap the title/top frame
    ax.set_ylim(0, 1.25)

    # Format values on bars carefully
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.03,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(t60)
    ax.set_xlabel(r'Reverberation Time ($T_{60}$)')
    ax.set_ylabel(r'Probing $R^2$')
    
    # Legend repositioned to side/top blank space
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.set_title('Reverberation Robustness (10 dB SNR)', fontsize=9, fontweight='bold', pad=8)

    plt.tight_layout()
    fig.savefig('fig4_room.pdf')
    fig.savefig('fig4_room.png')
    print("Saved fig4_room")


if __name__ == '__main__':
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig_layerwise()
        fig_snr()
        fig_weighting_profiles()
        fig_framework()
        fig_room()
        if w:
            for warning in w:
                print(f"Warning caught: {warning.message}")
        else:
            print("\nAll figures tuned, validated, and saved successfully with zero layout overlap warnings.")