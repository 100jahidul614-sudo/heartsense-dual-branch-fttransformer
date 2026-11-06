# Environment of the verified runs (5–6 November 2026)

| Component | Value |
| --- | --- |
| OS | Windows 11 (build 26300), 64-bit |
| CPU / RAM | Intel Core Ultra 9 275HX (24 threads) / 31.4 GB |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12 GB, compute capability 12.0 |
| NVIDIA driver | 591.91 (driver CUDA version 13.1) |
| Python | 3.12.10 (64-bit) |
| PyTorch | 2.9.1+cu128: CUDA 12.8 build, cuDNN 9.10.02 (91002). torchvision 0.24.1, torchaudio 2.9.1 |
| scikit-learn / NumPy / pandas / SciPy | 1.8.0 / 2.3.5 / 2.3.3 / 1.16.3 |
| LightGBM / imbalanced-learn / pytorch-tabnet | 4.6.0 / 0.14.1 / 4.1.0 |
| matplotlib / pyreadstat / openpyxl | 3.10.8 / 1.3.3 / 3.1.5 |
| Not installed | XGBoost, CatBoost. The original environment did not have them, so the baseline script skipped those two models. Installing them would add two models to Stage 3 |

- `requirements.txt` pins 47 package versions **captured from the project's `.venv312` interpreter during the reproduction** (read from its site-packages on 5 November 2026). The 48th package there is pip itself, pinned separately to 25.3 by `00_setup_env.ps1`.
- These are not verified historical versions: the versions installed for the original January 2026 experiments were not recorded. The verified runs reproduced the original saved outputs bit for bit with the captured versions.
- The verified runs set these process settings, none of which affects computation:
  - `MPLBACKEND=Agg`;
  - `PYTHONIOENCODING=utf-8`;
  - `PYTHONUNBUFFERED=1`;
  - `PYTHONDONTWRITEBYTECODE=1`.
- GPU code uses mixed precision. The proposed-model scripts set fixed seeds and deterministic cuDNN. On different hardware, drivers or library builds, results may differ in the last digits; the comparison tools report any such difference.

**Records:**
- `evidence/logs/stage0_env/`: `env_report.json`/`.txt`, `pip_freeze.txt`, `nvidia_smi.txt`, and the GPU and CPU reports.
- `environment.json` in each stage log folder: full package list, PyTorch/CUDA/cuDNN, GPU and driver.

**Install:**
- Windows: `00_setup_env.ps1`.
- Other systems: `pip install -r env/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`, then run `tools/check_env.py`.
