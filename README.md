# Cross-Power Is All You Need

Official code for **"What Do Neural Networks Learn for TDOA Estimation? A Cross-Architecture Probing Study"** (Interspeech 2026).

## Setup

```bash
git clone https://github.com/york1to/cross-power-is-all-you-need.git
cd cross-power-is-all-you-need
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Experiments

Each script runs standalone and writes `.pt` data and `.png` plots to `results/`:

| Script | Experiment |
|--------|-----------|
| `perfreq_probe.py` | Per-frequency probing (start here) |
| `architectures.py` | Cross-architecture: MLP, CNN, Transformer |
| `followup.py` | Matched-parameter MLP + nonlinear probe |
| `snr_sweep.py` | SNR sweep |
| `room_validation.py` | Reverberant rooms (pyroomacoustics) |
| `learned_weighting.py` | Gradient attribution / learned weighting |
| `freq_masking.py` | Causal frequency masking |
| `observed_phat_probe.py` | Observed-PHAT probing |
| `gcc_benchmark.py` | GCC weighting benchmark |
| `gcc_mismatch.py` | Cross-SNR mismatch |
| `ngcc_comparison.py` | NGCC preprocessing comparison |
| `locata_probing.py` | LOCATA real recordings |

LOCATA experiments require the [LOCATA dataset](https://www.locata.lms.tf.fau.de/) (not included). A CUDA GPU is recommended; CPU also works.

## Citation

```bibtex
@inproceedings{kang2026tdoa,
  title     = {What Do Neural Networks Learn for {TDOA} Estimation? A Cross-Architecture Probing Study},
  author    = {Kang, Yaozhong and Wang, Jiang and Shi, Runwu and Ashizawa, Takeshi and Yen, Benjamin and Nakadai, Kazuhiro},
  booktitle = {Proc. INTERSPEECH 2026},
  year      = {2026},
}
```

## License

[MIT](LICENSE)
