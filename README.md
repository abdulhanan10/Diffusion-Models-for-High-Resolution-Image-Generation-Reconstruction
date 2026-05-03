## Diffusion Models for High-Resolution Image Generation & Reconstruction

**National University of Computer and Emerging Sciences — Spring 2026**
**Students:** 22F-8762 | 22F-3275

---

## Overview

This notebook implements a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch in PyTorch for high-resolution face image generation and reconstruction. The model is trained on the CelebA-HQ dataset (or FFHQ as an alternative) and demonstrates the full diffusion pipeline — from forward noising to reverse denoising.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Results](#results)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Gradio App](#gradio-app)
8. [Requirements](#requirements)
9. [How to Run](#how-to-run)
10. [Troubleshooting](#troubleshooting)

---

## Environment Setup

- **Platform:** Kaggle Notebooks
- **GPU:** NVIDIA Tesla T4 × 2 (DataParallel)
- **Framework:** PyTorch with CUDA
- **Mixed Precision:** Enabled via `torch.cuda.amp.GradScaler`

---

## Dataset

### Primary — CelebA-HQ 256×256

| Property | Value |
|----------|-------|
| Kaggle Dataset | `celebahq256-images-only` |
| Path | `/kaggle/input/celebahq256-images-only/data256x256` |
| Resolution | 256 × 256 (resized to 128 × 128 for training) |

### Alternative — FFHQ Thumbnails

| Property | Value |
|----------|-------|
| Kaggle Dataset | `ffhq-face-data-set` |
| Path | `/kaggle/input/ffhq-face-data-set/thumbnails128x128` |
| Resolution | 128 × 128 (matches `IMG_SIZE` directly) |

> **Note:** To switch datasets, update `DATA_DIR` in the configuration cell (Cell 02).

### Preprocessing

- Resize to `128 × 128`
- Random horizontal flip (augmentation)
- Normalize to `[-1, 1]`

---

## Model Architecture

### U-Net

| Component | Details |
|-----------|---------|
| Encoder channels | 64 → 128 → 256 |
| Time embedding | Sinusoidal + MLP projection |
| Conditioning | Affine scale/shift per ResBlock |
| Skip connections | Yes (encoder → decoder) |

### Diffusion Scheduler

| Hyperparameter | Value |
|----------------|-------|
| Timesteps (T) | 400 |
| Noise schedule | Linear (β: 1e-4 → 0.02) |
| Loss | MSE on predicted vs. actual noise |

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Image size | 128 × 128 |
| Batch size | 16 |
| Epochs | 50 |
| Learning rate | 2e-4 |
| Optimizer | AdamW |
| LR Scheduler | Cosine Annealing |
| Gradient clipping | 1.0 |
| Mixed precision | ✅ |
| Multi-GPU | ✅ (DataParallel) |

---

## Results

Outputs are saved to `/kaggle/working/outputs/`:

| File | Description |
|------|-------------|
| `forward_steps.png` | Forward diffusion — 5 progressive noising steps |
| `reverse_steps.png` | Reverse diffusion — 5 denoising intermediates |
| `reconstruction.png` | Side-by-side original vs. reconstructed image |
| `generated_images.png` | Grid of purely generated images |
| `generated_1.png` … `generated_N.png` | Individual generated images |

---

## Evaluation Metrics

Quantitative evaluation is performed using:

- **PSNR** (Peak Signal-to-Noise Ratio) — measures pixel-level fidelity in dB
- **SSIM** (Structural Similarity Index) — measures perceptual/structural similarity

Metrics are computed for:
1. Reconstruction quality (noised → denoised vs. original)
2. Generated images vs. random dataset references

---

## Gradio App

An interactive **Gradio web app** is included for inference:

- **Controls:** Number of images (1–5), toggle intermediate denoising steps
- **Outputs:** Gallery of generated images + optional denoising step visualization
- **Launch:** `demo.launch(share=True)` — generates a public shareable link
- **Link:** "https://huggingface.co/spaces/abdilhanan01/Ass04"

---

## Requirements

```
torch
torchvision
gradio
scikit-image
matplotlib
Pillow
tqdm
numpy
```

> All packages are pre-installed on Kaggle except `gradio` and `scikit-image`, which are installed at runtime via `pip install -q gradio scikit-image`.

---

## How to Run

1. **Open** the notebook on [Kaggle](https://www.kaggle.com)
2. **Add the dataset** via the right sidebar → `+ Add Data` → search `celebahq256-images-only`
3. **Enable GPU** — Accelerator → GPU T4 × 2
4. **Run All** — `Run → Restart & Run All`
5. After training, the Gradio app will launch with a public share link

---

## Troubleshooting

### `FileNotFoundError: No images found under /kaggle/input/...`

The dataset is not attached to your notebook.

**Fix:** Go to the Kaggle sidebar → `+ Add Data` → search for `celebahq256-images-only` and click Add. Then re-run from Cell 02.

Alternatively, switch to FFHQ by changing Cell 02:
```python
DATA_DIR = '/kaggle/input/ffhq-face-data-set/thumbnails128x128'
```

### Verify attached datasets

```python
import os
print(os.listdir('/kaggle/input'))
```

---

## Notebook Structure

| Cell | Section |
|------|---------|
| 01 | Install dependencies |
| 02 | Imports & reproducibility seed |
| 03 | Hyperparameters & configuration |
| 04 | Data preprocessing (Part 1) |
| 05 | Forward diffusion process |
| 06 | U-Net architecture |
| 07 | Training setup |
| 08 | Training loop |
| 09 | Image reconstruction |
| 10 | Image generation |
| 11 | Visualization module |
| 12 | PSNR & SSIM evaluation |
| 13 | Gradio app deployment |

---

## Summary

| Item | Detail |
|------|--------|
| Dataset | CelebA-HQ 256 × 256 |
| Image Resolution | 128 × 128 |
| Diffusion Timesteps | 400 |
| Noise Schedule | Linear (β: 1e-4 → 0.02) |
| Architecture | U-Net — 64 → 128 → 256 channels |
| Time Embedding | Sinusoidal + MLP |
| Loss | MSE on predicted vs. actual noise |
| Optimizer | AdamW (lr = 2e-4) |
| Mixed Precision | ✅ |
| Multi-GPU | ✅ DataParallel |
| Evaluation | PSNR & SSIM |
| Deployment | Gradio |
