# Group F
Project repository for the Computational Imaging (2025–26, University of Bologna,
proff. Loli Piccolomini & Evangelista) group assignment.

The project studies **deblurring and denoising of retinal images** on a fundus
retina dataset, formulated as the inverse problem `y = A x + e`, where `A` is a
known motion-blur operator and `e` is additive Gaussian noise. The required setup
follows the official project specifications:

- work on grayscale images resized to 256×256, with values in `[0, 1]`;
- generate degraded observations with a **motion-blur** operator (angle 45°, kernel size 9);
- add **Gaussian noise at four levels**: σ ∈ {0.005, 0.01, 0.05, 0.1};
- compare all methods on the **same degraded inputs**;
- report PSNR, SSIM, visual comparisons, and discussion.

"Deblur + Denoise" is a single task: deblurring with a controlled amount of added
noise (`y = A x + e`), **not** two independent steps.
## Presentation
```text
https://canva.link/4xcfmijl1udqfwn
```
## Methods

This is a three-student group, so the repository covers four methods:

1. **Variational** — FISTA with a wavelet sparsity prior (`R(x) = ‖W x‖₁`, Daubechies db4);
2. **Hybrid** — Adaptive Weighted Total Variation (`R(x) = ‖w ⊙ |∇x|‖₁`),
   following Morotti et al. (2025), arXiv:2501.09845;
3. **End-to-end** — a UNet reconstructor trained per noise level;
4. **Generative** — DiffPIR, a plug-and-play diffusion prior adapted to this inverse problem.

## Shared data (important)

For a fair comparison, all methods are evaluated on **identical inputs**: a single
set of pre-computed `.pt` files generated once with the **IPPy** library
(`operators.Blurring(kernel_type="motion", kernel_size=9, motion_angle=45)` plus
`utilities.gaussian_noise`). The noise model is IPPy's relative one
(`‖e‖ = noise_level · ‖y‖`).

- Deterministic selection: the first 4000 images in alphabetical order.
- Split 70/15/15 with `torch.Generator().manual_seed(0)` → train 2800, validation 600, test 600.
- Files: `retina_processed/{train,validation,test}/noise_<level>.pt`, each a dict with
  `clean`, `degraded` (tensors `[N,1,256,256]` in `[0,1]`), `source_paths`, `metadata`.
- All final metrics are averaged over the **600 test images**, per noise level.

## Dataset
The notebooks `00_data_degradation.ipynb`, `01_fista_wavelet.ipynb`, `02_UNet.ipynb`, `03_generativeDiffPir.ipynb`and `04_adaptive_wtv.ipynb` use following dataset:
```text
https://drive.google.com/drive/folders/1RaNytO0PolkW90Y5GBX1eqH7Z3iIKqXJ?usp=sharing
```
## Repository layout

```text
GroupF/
├── README.md
├── Notebooks/                      # one notebook per method (self-contained)
│   ├── 01_fista_wavelet.ipynb      # Variational  — FISTA + Wavelet
│   ├── 02_UNet.ipynb               # End-to-end   — UNet
│   ├── 03_generativeDiffPir.ipynb  # Generative   — DiffPIR
│   └── 04_adaptive_wtv.ipynb       # Hybrid       — Adaptive Weighted TV
└── Output/                         # generated metrics and figures
    ├── Fista/                      # fista_all.csv, fista_tuning.csv, plot.png
    ├── Hybrid/                     # awtv_all.csv, plot.png
    ├── UNet/                       # summary metrics, visual grids
    └── DiffPIR/                    # metrics and visual results
```

Large data, trained weights, and the original dataset are **not tracked by Git**.

## External layout

The notebooks expect the shared processed data and the IPPy library to be available.
The expected structure on Google Drive is:

```text
MyDrive/
└── GroupF/
    └── retina_processed/    # shared .pt files (generated once), outside Git
        ├── train/
        ├── validation/
        └── test/
```

The test split is fixed by the seed-0 70/15/15 partition described above; the
validation subset is used only for choosing the regularization parameter λ
(and τ for DiffPIR).

## Setup

The variational and hybrid methods run on **CPU** (NumPy); the UNet and DiffPIR
benefit from a GPU.

Install IPPy (operator + noise model):

```bash
pip install git+https://github.com/devangelista2/IPPy
```

On Colab, mount Google Drive and let the notebooks auto-locate the shared data:

```python
from google.colab import drive
drive.mount('/content/drive')

import glob, os
from pathlib import Path
hits = glob.glob("/content/drive/MyDrive/**/noise_*.pt", recursive=True)
PROCESSED = Path(os.path.dirname(os.path.dirname(hits[0])))   # -> .../GroupF/retina_processed
```

If the shared `retina_processed` folder is shared with you rather than owned,
add a shortcut to it inside *My Drive* so the recursive search above can find it.

## Members

- Jacopo Bonifazi — 0001217841
- Gaetano Muscarello — 0001218184
- Shivam Kumar — 0001242586

## References

- E. Morotti, D. Evangelista, A. Sebastiani, E. Loli Piccolomini (2025).
  *Adaptive Weighted Total Variation boosted by learning techniques in few-view
  tomographic imaging.* arXiv:2501.09845.
- A. Beck, M. Teboulle (2009). *A Fast Iterative Shrinkage-Thresholding Algorithm
  for Linear Inverse Problems.* SIAM J. Imaging Sci.
- L. Rudin, S. Osher, E. Fatemi (1992). *Nonlinear total variation based noise
  removal algorithms.*
- IPPy library: https://github.com/devangelista2/IPPy
