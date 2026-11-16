# gsplat on AMD ROCm (gfx1250 / MI45x) — Build & Run

End-to-end setup for building and benchmarking this repo
(`AMD-Ecosystem/gsplat`, branch `sync/mi45x_support`) on AMD `gfx1250` A0 silicon
with the ROCm pip ("whl-next") stack.

> **The single most important rule:** create the Python venv **OUTSIDE** this
> repository. If the ROCm SDK ends up under the repo (the build directory),
> PyTorch's hipify rewrites the SDK headers in place and the HIP build dies with
> thrust/hipcub `redefinition` errors. See Troubleshooting #2.

---

## Verified working build

The exact stack below was confirmed end-to-end on this hardware: a full
30,000-step Mip-NeRF 360 `stump` training run completed, including densification
(which exercises hipBLASLt GEMM) — PSNR 26.70 / SSIM 0.777 / LPIPS 0.146 at
~96 steps/s, render ~2.4 ms/image.

| Component | Verified value |
| --- | --- |
| ROCm / pip tag | `10.3.0a20261105` (nightly `whl-next`) |
| PyTorch | `2.14.0+rocm10.3.0a20261105` |
| HIP runtime | `7.18.26441` |
| gsplat | `1.5.3` (built from source for `gfx1250`, wave32) |
| Python | 3.12 |
| amdgpu / KFD driver | `7.1.0.31300009` |
| GPU | `gfx1250` A0 (wave32); reports as "AMD Radeon Graphics" |
| Required runtime flag | `HSA_DISABLE_GFX12_STRICT=1` |

With `HSA_DISABLE_GFX12_STRICT=1` the device enumerates as plain `gfx1250`,
torch's device kernels load, and hipBLASLt GEMMs resolve against the A0
`gfx1250v0` Tensile library — no `TORCH_BLAS_PREFER_HIPBLASLT=0` needed.

The `10.2.0a20261019` stack (torch `2.14.0+rocm10.2.0a20261019`, HIP
`7.17.26413`) was also tested: general/elementwise kernels run and the device
enumerates as plain `gfx1250`, but **every hipBLASLt GEMM fails** — its
`gfx1250v0/` directory ships no `TensileLibrary_lazy_gfx1250.dat` master index,
so training dies at the first densification `einsum` with
`HIPBLAS_STATUS_INVALID_VALUE`. Use 10.3.0a instead.

---

## 0. Get the repo

```bash
git clone --branch sync/mi45x_support https://github.com/AMD-Ecosystem/gsplat.git
cd gsplat
git submodule update --init --recursive   # pulls gsplat/cuda/csrc/third_party/glm
```

---

## 1. Create a venv OUTSIDE the repo

```bash
python3.12 -m venv ~/venvs/gsplat-eco
source ~/venvs/gsplat-eco/bin/activate
```

Verify it is not under the repo:

```bash
python -c "import sys; print(sys.prefix)"   # must NOT be inside the gsplat checkout
```

---

## 2. Install the ROCm + PyTorch stack

Pick ONE of the two stacks below. Keep **every** component on the same ROCm tag
(torch, torchvision, the SDK, and both `amd-*-device-gfx1250` packages). Mixing
tags (e.g. torch `rocm10.1.0` with `rocm-sdk-devel 10.2.0a`) causes subtle
toolchain/runtime breakage.

### Option A — Stable 10.1.0

```bash
python -m pip install -i https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1250]==2.14.0+rocm10.1.0" \
  "rocm[libraries,devel,device-gfx1250]==10.1.0" \
  "torchvision==0.29.0a0+rocm10.1.0" \
  "amd-torchvision-device-gfx1250==0.29.0a0+rocm10.1.0"
```

### Option B — Nightly 10.3.0a (helper script, RECOMMENDED)

Use the provided `install_rocm_nightly.sh` (it refuses to run unless a venv OR
conda env is active AND that env is outside the repo):

```bash
./install_rocm_nightly.sh
```

It installs, all pinned to `10.3.0a20261105` from
`https://nightly.repo.amd.com/rocm/whl-next/`:
`rocm[libraries,devel]`, `rocm-sdk-device-gfx1250`, `torch`,
`amd-torch-device-gfx1250`, `torchvision`, `amd-torchvision-device-gfx1250`.

> **Why 10.3.0a over 10.2.0a:** 10.2.0a shipped no gfx1250v0 Tensile library, so
> every hipBLASLt GEMM on A0 crashed. 10.3.0a fixes that (the
> `gfx1250v0/TensileLibrary_lazy_gfx1250.dat` master index now ships), so GEMMs
> run through hipBLASLt with **no** `TORCH_BLAS_PREFER_HIPBLASLT=0` needed.
> 10.3.0a does, however, enumerate the A0 device as `gfx1250-strict` by default,
> which makes kernel loads fail with `invalid kernel file` — so you must export
> **`HSA_DISABLE_GFX12_STRICT=1`** at runtime (see step 5 and Troubleshooting #5).

### Why the `amd-*-device-gfx1250` and ROCm torchvision packages are required

- `torch` / `torchvision` base wheels do **not** contain gfx1250 device kernels;
  `amd-torch-device-gfx1250` / `amd-torchvision-device-gfx1250` provide them.
- The stock PyPI `torchvision` (e.g. `0.29.2`) is **not** ROCm-enabled and fails
  at import with `operator torchvision::nms does not exist`. You must use the
  `+rocmXX` torchvision wheel. `torchmetrics` pulls torchvision in transitively,
  so the trainer needs it even though it never imports torchvision directly.
- `torchaudio` is **not** needed by gsplat or the benchmark.

Sanity check:

```bash
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "hip", torch.version.hip)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
PY
```

---

## 3. Build gsplat from source

```bash
cd /path/to/gsplat          # repo root
source ./path.sh            # exports ROCM_PATH/HIP_PATH and header include paths
export PYTORCH_ROCM_ARCH=gfx1250
rm -rf build gsplat/hip     # gsplat/hip is hipify-generated; always start clean
python setup.py bdist_wheel
pip install dist/amd_gsplat-*.whl
```

Notes:
- `gsplat/hip/` is git-ignored and regenerated by hipify on every build — never
  hand-edit it; edit `gsplat/cuda/...` instead.
- The build auto-selects wave width from the arch: `gfx10xx/11xx/12xx` → wave32
  (`-DGSPLAT_WARP_SIZE=32`), CDNA (`gfx9xx`) → wave64.

---

## 4. Get the dataset (Mip-NeRF 360 "stump")

The scene is only distributed inside the full `360_v2.zip`; there is no
per-scene download. To save space, extract just `stump` (the smallest scene,
125 images):

```bash
cd /path/to/gsplat/examples
mkdir -p data/360_v2
curl -L -o /tmp/360_v2.zip http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip /tmp/360_v2.zip "stump/*" -d data/360_v2
rm /tmp/360_v2.zip
# verify the COLMAP model exists:
ls data/360_v2/stump/sparse/0
```

(Full set: `python datasets/download_dataset.py --dataset mipnerf360`.)

---

## 5. Run

Set the AMD runtime knob for your stack, then launch.

```bash
# --- ROCm 10.3.0a (recommended) ---------------------------------------------
# A0 enumerates as "gfx1250-strict" by default -> kernel loads fail with
# "invalid kernel file". This flag makes it enumerate as plain gfx1250 and lets
# hipBLASLt GEMMs run (no TORCH_BLAS_PREFER_HIPBLASLT needed). See Troubleshooting #5.
export HSA_DISABLE_GFX12_STRICT=1

# --- ROCm 10.1.0 / 10.2.0a (older) ------------------------------------------
# These have the hipBLASLt gfx1250v0 packaging bug; route GEMMs through rocBLAS:
#   export TORCH_BLAS_PREFER_HIPBLASLT=0

# Example: single-scene training (examples/simple_trainer.py)
cd /path/to/gsplat/examples
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
  --data_dir data/360_v2/stump/ --data_factor 4 \
  --result_dir ./results/stump
```

## 6. Benchmark (from `examples/`)

The repo's own benchmark suite lives in `examples/benchmarks/`. `basic.sh`
trains + evals every Mip-NeRF 360 scene (`garden bicycle stump bonsai counter
kitchen room`) and prints each scene's `val*`/`train*` stats JSONs (PSNR/SSIM/
LPIPS, render time, Gaussian count, VRAM):

```bash
cd /path/to/gsplat/examples
bash benchmarks/basic.sh
```

To benchmark a single scene (e.g. the smallest, `stump`), run the same command
`basic.sh` uses for it:

```bash
cd /path/to/gsplat/examples
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
  --disable_viewer --data_factor 4 --render_traj_path ellipse \
  --data_dir data/360_v2/stump/ --result_dir ./results/stump
# metrics: results/stump/stats/val_step29999.json (psnr/ssim/lpips/render time)
#          results/stump/stats/train_step29999_rank0.json (train time/mem/num_GS)
```

### GPU kernel timing (`PROFILE` / `TILE_SIZE`)

`examples/simple_trainer.py` reads two optional env vars, so kernel-level perf
work needs no code changes:

- `TILE_SIZE` — rasterizer tile size (e.g. `8` or `16`), applied to train + eval.
- `PROFILE=1` — wraps a short window of steps in `torch.profiler` and dumps a
  Chrome trace plus a top-kernel table (`kernel_summary.txt`, sorted by self
  CUDA/HIP time) under `<result_dir>/profiler/`. Window via `PROFILE_START`
  (default 1000) / `PROFILE_STEPS` (default 5).

```bash
cd /path/to/gsplat/examples
PROFILE=1 PROFILE_START=21100 PROFILE_STEPS=3 TILE_SIZE=8 \
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
  --disable_viewer --data_factor 4 --render_traj_path ellipse \
  --data_dir data/360_v2/stump/ --result_dir ./results/stump
# kernel table: results/stump/profiler/kernel_summary.txt
# chrome trace: results/stump/profiler/trace_step21100_21102.json
```

> Compare kernels across runs using the **`CUDA time avg`** (per-call) column,
> not the `Self CUDA` total — the latter scales with `PROFILE_STEPS` (call
> count). Also profile at the **same** `PROFILE_START`, since per-step cost grows
> with the Gaussian count as densification proceeds.

### Tile size / which wave32 kernel runs

- `TILE_SIZE` is passed straight to gsplat's `rasterization(..., tile_size=...)`;
  on this build the default is **8**.
- `TILE_SIZE=8` (`block_size == 64`, `CDIM <= 16`) dispatches the specialized
  zero-LDS **multi-tile** kernel
  `rasterize_bs32_multitile_to_pixels_3dgs_bwd_kernel`.
- `TILE_SIZE=16` (256-thread block) dispatches the **generic** multi-warp kernel
  `rasterize_to_pixels_3dgs_bwd_kernel` (more occupancy).
- Which is faster is hardware dependent — A/B both and compare the per-call
  `rasterize_to_pixels_3dgs_bwd` time (the `CUDA time avg` column).

---

## Troubleshooting (errors seen on this hardware)

**1. `operator torchvision::nms does not exist`**
Stock PyPI torchvision installed instead of the ROCm build. Reinstall the
`torchvision==<ver>+rocmXX` wheel (and `amd-torchvision-device-gfx1250`) matching
your torch tag. See step 2.

**2. hipcub/thrust `redefinition of 'is_non_bool_arithmetic'` (and friends) during `setup.py`**
The venv is **inside** the repo, so hipify hipified the ROCm SDK headers in place
(generating `*_hip.h` twins that collide with the originals). Fix: delete the
venv, recreate it OUTSIDE the repo (step 1), reinstall (step 2), `rm -rf build
gsplat/hip`, rebuild. The real sources in `gsplat/cuda/include/` are already
correct; editing `gsplat/hip/` does nothing (it's regenerated).

**3. `COLMAP directory data/360_v2/stump/sparse does not exist`**
Dataset not downloaded, or wrong CWD. `--data_dir` is relative to `examples/`.
See step 4.

**4. `mkdir: cannot create directory '.../results/...': Permission denied`**
The `results/` dir is owned by root from a prior run. Take ownership:
`sudo chown -R "$USER":"$USER" /path/to/gsplat/examples/results`.

**5. hipBLASLt / GEMM failure — behaviour depends on the ROCm stack**

- **10.1.0:** `hipModuleLoad failed: .../Kernels.so-000-gfx1250-xnack-.hsaco: file not found`
- **10.2.0a:** `rocblaslt error: gfx1250 device with asicRevision 0 (A0) needs the
  gfx1250v0 GEMM library, but ".../gfx1250v0" holds no Tensile library`, ending in
  `HIPBLAS_STATUS_INVALID_VALUE` from `hipblasLtMatmulAlgoGetHeuristic`. Even a
  bare `a @ b` on GPU crashes. Root cause: the `gfx1250v0/` subtree ships kernel
  objects but no `TensileLibrary_lazy_gfx1250.dat` master index, and the strict A0
  guard refuses the generic `gfx1250/` kernels.
  **Workaround (10.1/10.2 only):** route torch GEMMs through rocBLAS:
  ```bash
  export TORCH_BLAS_PREFER_HIPBLASLT=0
  ```
- **10.3.0a:** the `gfx1250v0` Tensile library now ships (hipBLASLt bug fixed),
  but the device enumerates as `gfx1250-strict` by default and kernel loads fail
  with `CUDA error: invalid kernel file` (`hipErrorInvalidKernelFile`) — the
  shipped kernel objects aren't built for the strict ISA.
  **Workaround (10.3.0a):**
  ```bash
  export HSA_DISABLE_GFX12_STRICT=1   # device -> plain gfx1250; GEMMs work via hipBLASLt
  ```
  No `TORCH_BLAS_PREFER_HIPBLASLT=0` is needed on 10.3.0a.

Caveat: `HSA_DISABLE_GFX12_STRICT=1` makes hipBLASLt use the generic `gfx1250`
library rather than the A0-specific `gfx1250v0` one, so A0 numerics remain
provisional. Proper fix: a 10.3.0a build whose `gfx1250-strict` kernel objects
load cleanly so the A0-correct `gfx1250v0` path can be used without the flag.

---

## Quick reference

| Item | Value |
| --- | --- |
| Repo / branch | `AMD-Ecosystem/gsplat` / `sync/mi45x_support` |
| GPU arch | `gfx1250` (A0), wave32 |
| venv/conda env | **outside** the repo (e.g. `~/venvs/gsplat-eco`) |
| ROCm stack | 10.3.0a20261105 (nightly, recommended) / 10.1.0 (stable) / 10.2.0a (older) — keep tags consistent |
| Build | `source ./path.sh && export PYTORCH_ROCM_ARCH=gfx1250 && python setup.py bdist_wheel` |
| Runtime env | 10.3.0a: `HSA_DISABLE_GFX12_STRICT=1` · 10.1/10.2: `TORCH_BLAS_PREFER_HIPBLASLT=0` · optional `TILE_SIZE=8` |
