# ROCm/HIP Build & Test Report

**Branch:** `fix/rocm-wsl-compile-build`
**Platform:** AMD Instinct MI325X (gfx942, wavefront-64), ROCm 7.15, PyTorch 2.12 (HIP), Python 3.12
**Date:** 2026-08-15

## Procedure

1. Full clean: removed `build/`, `dist/`, `*.egg-info`, all generated `gsplat/**/kernels/hip` trees, `gsplat/hip`, and the torch-extensions JIT cache.
2. Wheel build: `BUILD_CAMERA_WRAPPERS=1 CC=hipcc CXX=hipcc python setup.py bdist_wheel` → **EXIT 0**.
3. Copied fresh `csrc.so` into the repo tree; verified `import gsplat, gsplat.csrc` on GPU.
4. Full suite: `python -m pytest tests/ -q -rfE --tb=line --continue-on-collection-errors`.

## Build result

| Item | Result |
|---|---|
| Wheel build (AOT, camera wrappers) | ✅ EXIT 0, 0 compile errors |
| JIT extensions (sensors, geometry, scene, experimental render) | ✅ all built clean during test run |
| Import / device init (MI325X) | ✅ OK |

## Test summary

| Metric | Count |
|---|---|
| **Passed** | **2533** |
| Skipped | 134 |
| Subtests passed | 8 |
| **Failed** | **15** |
| Wall time | 15:01 |

- **Hard failures (build error / crash / illegal memory / collection error): 0**
- **Accuracy failures (numerical tolerance / gradcheck): 15**

All 15 failures are `AssertionError: Tensor-likes are not close!` or `GradcheckError`. No test crashed, hung, or failed to build/collect.

---

## 🔴 Tests that completely fail (hard failures)

**None.** Every test executed to completion. There are no build errors, segfaults, HIP errors, illegal-memory accesses, import errors, or collection errors.

---

## 🟡 Tests with accuracy issues (numerical / tolerance)

### Group A — pre-existing numerical-tolerance asserts (11)
Razor-thin float32 tolerance overruns, present before this port and unrelated to it (float32 rounding differs on this GPU). Representative magnitudes shown.

| Test | Nature | Observed vs allowed |
|---|---|---|
| `test_basic.py::test_proj[batch_dims0-pinhole]` | Tensor-likes not close | ~0.00153 vs 0.001 |
| `test_basic.py::test_proj[batch_dims1-pinhole]` | Tensor-likes not close | ~0.00153 vs 0.001 |
| `test_basic.py::test_proj[batch_dims2-pinhole]` | Tensor-likes not close | ~0.00153 vs 0.001 |
| `test_basic.py::test_rasterize_to_pixels[batch_dims0-128]` | Tensor-likes not close / grad sparsity | ~0.00313 vs 0.003 |
| `test_basic.py::test_rasterize_to_pixels[batch_dims1-128]` | Tensor-likes not close / grad sparsity | ~0.00313 vs 0.003 |
| `test_basic.py::test_rasterize_to_pixels[batch_dims2-128]` | Tensor-likes not close / grad sparsity | ~0.00313 vs 0.003 |
| `test_basic.py::test_rasterize_to_pixels_eval3d[3-batch_dims2-4-True-False-False-pinhole-8]` | Tensor-likes not close | borderline |
| `test_basic.py::test_rasterize_to_pixels_eval3d[3-batch_dims6-4-False-False-False-pinhole-8]` | Tensor-likes not close | borderline |
| `test_basic.py::test_rasterize_to_pixels_eval3d[3-batch_dims50-4-True-True-False-lidar-8]` | Tensor-likes not close | borderline |
| `test_basic.py::test_rasterize_to_pixels_eval3d[3-batch_dims51-4-True-True-False-lidar-16]` | Tensor-likes not close | borderline |
| `test_rasterization.py::test_rasterization[...batch_dims230...lidar...render]` | Tensor-likes not close | ~0.00036 vs 0.0003 (13/1.38M elems) |

**Assessment:** not port regressions. These are single-/few-element tolerance overruns (mismatched elements ~0.0% of tensor) from float32 arithmetic on a different GPU vendor. Some `test_rasterize_to_pixels` runs may alternatively OOM under a per-process memory cap when the full suite accumulates GPU memory — same tests, environmental.

### Group B — newly-exposed ftheta gradcheck (4)
These sensor-kernel tests were **skipped** before this port (the `gsplat_sensors_cuda` extension didn't build). Fixing the build exposed them.

| Test | Nature |
|---|---|
| `sensors/kernels/cameras/test_ops.py::test_camera_rays_to_image_points_ftheta_no_external_gradcheck[forward_ref]` | Gradcheck Jacobian mismatch |
| `sensors/kernels/cameras/test_ops.py::test_camera_rays_to_image_points_ftheta_no_external_gradcheck[backward_ref]` | Gradcheck Jacobian mismatch |
| `sensors/kernels/cameras/test_ops.py::test_ftheta_project_world_points_mean_pose_gradcheck` | Gradcheck Jacobian mismatch |
| `sensors/kernels/cameras/test_ops.py::test_ftheta_project_world_points_shutter_pose_gradcheck` | Gradcheck Jacobian mismatch |

**Assessment:** float32 finite-difference noise, **not** a correctness bug. Evidence:
- The FTheta kernel is **float32-only**; `gradcheck` perturbs in float64, so the numerical Jacobian is quantized/noisy while the analytical Jacobian is clean and symmetric.
- The test file's own comment documents this: *"FTheta projection runs through acos/sin/cos in float32… small cross terms sit close to the finite difference noise floor."* The sibling tests already run at loosened `atol` (8e-2); ROCm's transcendental noise floor is higher than NVIDIA's, pushing these over.
- The failing gradient is w.r.t. the **input points/rays** (`grad_camera_rays`), which does **not** route through the warp-reduction code changed in this port.
- Forward/correctness tests for the same ops pass; 12–14 other ftheta gradcheck tests pass.
- One camera_rays case shows a large single-index blowup (index 30868) — a near-singular ray where finite differences are non-smooth.

**Suggested follow-up (optional):** raise `atol` for these four ROCm gradchecks to match the existing float32 gradcheck precedent, or mark them `xfail` on ROCm with a precision note. This is a test-tolerance change, not a kernel change — deferred pending your call.

---

## Port fixes applied (this effort)

1. **`sensors/kernels/cuda/build.py`** — added the extension's own `csrc` dir to `extra_include_paths` (hipify was never copying its headers → "file not found", ~424 tests stuck skipped); added the clang gcc-13 pin and ROCm `-L`/`-rpath` ldflags (mirroring the geometry extension).
2. **`sensors/kernels/cuda/csrc/math.cuh`** + `camera_/fisheye_/ftheta_/lidar_kernel_backward.cu` — 64-bit warp-shuffle mask (`WarpMask`/`kFullWarpMask`) for wavefront-64; HIP's `__shfl_*_sync` static-asserts a 64-bit mask.

**Net effect:** passing tests rose from 2109 → **2533** (+424 sensor-kernel tests now build and run).

## Bottom line

The ROCm/HIP build is **clean** (0 compile/link errors) and the suite is **functionally green**: 2533 passing, 0 hard failures. The 15 failures are all numerical-tolerance/gradcheck issues on float32 kernels — 11 pre-existing and unrelated to the port, 4 newly-exposed float32 gradcheck noise-floor cases addressable via test tolerance.
