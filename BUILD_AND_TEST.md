# ROCm/HIP Build & Test — Reproducible Recipe

Exact, copy-pasteable commands to clean, build (including all experimental
features), and test `aioss-gsplat` on AMD ROCm. Verified on the environment below.

## Verified environment

| Component | Value |
|---|---|
| GPU | AMD Instinct MI325X (gfx942, wavefront-64) |
| ROCm / HIP | 7.15.0 |
| PyTorch | 2.12.0+rocm7.15 (HIP build) |
| Python | 3.12.3 |
| Compiler | `hipcc` (AMD clang 23.0.0) from the pip `_rocm_sdk_devel` package |
| Host GCC for clang pin | `/usr/lib/gcc/x86_64-linux-gnu/13` (12/13/14 present; 13 chosen) |

> The build pins ROCm's clang to GCC 13 via `--gcc-install-dir` because GCC 14's
> libstdc++ `<format>`/`bits/unicode.h` breaks clang. Ensure GCC 12 or 13 is
> installed (`sudo apt-get install g++-13`).

---

## Install PyTorch and ROCm 10.1

Create a virtual environment, then install the stable ROCm 10.1 torch wheel
and the matching SDK (`hipcc` comes from the `devel` extra). `device-gfx942`
is MI300X / MI325X; swap it for another `[device-*]` extra if needed.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "torch[device-gfx942]==2.14.0+rocm10.1.0" \
    "rocm[libraries,devel,device-gfx942]==10.1.0"
```

---

## Step 0 — Activate environment

Every shell that builds or tests must have the venv **and** the ROCm env sourced.

```bash
cd /path/to/aioss-gsplat
source .venv/bin/activate          # your Python venv with torch-rocm installed
source ./path.sh                   # exports ROCM_PATH/HIP_PATH + headers onto CPATH
```

`path.sh` is required — without it, JIT builds fail with `hip/hip_runtime.h: No such
file or directory`. It sets `ROCM_PATH`, `HIP_PATH`, device-lib paths, and puts ROCm
headers on `CPATH`/`CPLUS_INCLUDE_PATH`.

Optional but recommended (silences an all-archs warning and speeds compiles):

```bash
export PYTORCH_ROCM_ARCH=gfx942
```

Sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
which hipcc && hipcc --version | head -2
```

---

## Step 1 — Clean everything

Removes build artifacts, **all generated `hip/` hipify trees**, and the torch JIT
cache. Deleting the `hip/` trees is mandatory: hipify skips files that already
exist, so a stale tree causes phantom "file not found" errors and ignores source
edits.

```bash
rm -rf build dist *.egg-info
rm -rf gsplat/hip \
       gsplat/experimental/render/kernels/hip \
       gsplat/geometry/kernels/hip \
       gsplat/scene/kernels/hip \
       gsplat/sensors/kernels/hip
rm -rf ~/.cache/torch_extensions/*
```

Verify clean:

```bash
ls build dist 2>/dev/null || echo "build/dist gone"
find gsplat -type d -name hip -path '*/kernels/*' ; echo "(no lines above = clean)"
```

---

## Step 2 — Full AOT wheel build (with camera wrappers)

Builds the main `gsplat/csrc.so` extension **including camera wrappers**. Must use
`CC=hipcc CXX=hipcc` (plain g++ can't parse the vendored GLM `__device__` code under
`-D__HIP__`).

```bash
BUILD_CAMERA_WRAPPERS=1 CC=hipcc CXX=hipcc python setup.py bdist_wheel
```

Expected: `[N/N]` compile steps with **0 errors**, ending in a `dist/amd_gsplat-*.whl`.

Install the wheel and copy the fresh `.so` into the repo tree (pytest uses
`pythonpath=.`, so it imports the repo copy):

```bash
pip install --force-reinstall --no-deps dist/amd_gsplat-*.whl
cp build/lib.linux-x86_64-cpython-312/gsplat/csrc.so gsplat/csrc.so
```

Verify import:

```bash
python -c "import gsplat, gsplat.csrc, torch; print('import OK', torch.cuda.get_device_name(0))"
```

> **Note (cupy detection / other servers):** older `setup.py` revisions crash in
> `_detect_cupy_requirement()` with `NameError: name 'cuda_roots' is not defined`.
> This branch fixes that and also skips CUDA-cupy detection entirely on ROCm. If you
> are on a checkout that still has the bug, bypass it with:
> `CUPY_PACKAGE=cupy BUILD_CAMERA_WRAPPERS=1 CC=hipcc CXX=hipcc python setup.py bdist_wheel`

---

## Step 3 — Build all experimental / JIT extensions

These are JIT-compiled on first import; build them explicitly to confirm they
compile clean before testing. Covers **geometry, sensors, scene, and the
experimental render inference** extension.

```bash
VERBOSE=1 MAX_JOBS=10 python - <<'PYEOF'
import sys; sys.path.insert(0, '.')
def run(name, fn):
    try:
        fn(); print(f"{name}: OK")
    except Exception as e:
        print(f"{name}: FAIL: {type(e).__name__}: {str(e)[:200]}")

from gsplat.geometry.kernels.cuda.build import build_and_load_geometry_cuda
run("geometry", build_and_load_geometry_cuda)

from gsplat.sensors.kernels.cuda.build import build_and_load_sensors_cuda
run("sensors", build_and_load_sensors_cuda)

from gsplat.scene.kernels._backend import _get_backend
run("scene", _get_backend)

from gsplat.experimental.render.kernels.cuda.build import (
    build_and_load_experimental_gaussian_render_inference_scene as exp,
)
run("experimental_render_inference", exp)
PYEOF
```

Expected output:

```
geometry: OK
sensors: OK
scene: OK
experimental_render_inference: OK
```

---

## Step 4 — Full test run

```bash
python -m pytest tests/ -q -rfE --tb=line --continue-on-collection-errors 2>&1 | tee /tmp/pytest.log
```

- `--continue-on-collection-errors` avoids a transient first-build collection abort.
- Runtime ≈ 15 min once extensions are cached.

Quick triage of the log:

```bash
# Hard failures (should be none): build/link/crash/collection
grep -cE 'Segmentation|core dumped|HIP error|illegal memory|error collecting|ImportError|ModuleNotFound|internal compiler|file not found' /tmp/pytest.log

# Failure summary line + list
grep -E '[0-9]+ (failed|passed)' /tmp/pytest.log | tail -1
grep -E '^FAILED' /tmp/pytest.log
```

---

## Expected result (this branch, MI325X)

- Build: wheel + all JIT/experimental extensions compile with **0 errors**.
- Tests: **~2533 passed, ~134 skipped, ~15 failed**.
- **0 hard failures.** All ~15 failures are numerical-tolerance / gradcheck asserts
  on float32 kernels (see `TEST_REPORT.md` for the categorized breakdown):
  - 11 pre-existing float32 tolerance overruns (`test_basic` proj/rasterize,
    one `test_rasterization` lidar), unrelated to the port;
  - 4 ftheta gradcheck cases (float32 finite-difference noise floor).

---

## Key gotchas (why each step matters)

| Gotcha | Consequence if skipped | Guard |
|---|---|---|
| `source ./path.sh` | `hip/hip_runtime.h not found` at JIT time | Step 0 |
| Delete `hip/` trees | Phantom "file not found"; source edits ignored | Step 1 |
| `CC=hipcc CXX=hipcc` | g++ chokes on GLM `__device__` under `-D__HIP__` | Step 2 |
| GCC 13 pin (`--gcc-install-dir`) | clang errors in GCC 14 `<format>`/`bits/unicode.h` | in `build.py` |
| Copy `csrc.so` into repo | pytest (`pythonpath=.`) imports stale extension | Step 2 |
| 64-bit warp mask | `__shfl_*_sync` static-assert on wavefront-64 | in kernels |
| Own `csrc` on include path | hipify never copies extension headers → skips | in `build.py` |

See `CUDA_TO_HIP_PORTING_GUIDE.md` for the project-agnostic version of these.
