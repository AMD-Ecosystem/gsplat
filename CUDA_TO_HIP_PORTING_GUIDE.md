# CUDA → HIP / ROCm Porting Guide

A project-agnostic playbook for porting a CUDA C++/PyTorch-extension codebase to
AMD ROCm (HIP), distilled from a real port to MI300X/MI325X (gfx942, wavefront-64,
ROCm 7.x, PyTorch HIP build, Python 3.12).

The advice below is ordered by how often it bites. Each item lists the **symptom**
(what the compiler/test shows), the **root cause**, and the **fix**.

---

## 0. Mental model & first principles

- **HIP is source-compatible with CUDA, not identical.** Most `__global__`,
  `__device__`, `<<<>>>`, `cudaMalloc`, etc. map 1:1. The failures cluster in a few
  areas: warp width, warp-sync intrinsics, half/bf16 intrinsics, vendor PTX
  helpers, `cub`, the toolchain/host-compiler, and the hipify build plumbing.
- **Two compile paths exist and behave differently:**
  1. **AOT wheel build** (`setup.py` via `CUDAExtension`/`hipify`) — the artifact
     shipped and usually what tests import.
  2. **JIT build** (`torch.utils.cpp_extension.load`) — per-extension, hipifies at
     import time into a generated `hip/` tree.
  Get **both** right; they have separate flag plumbing and separate failure modes.
- **Wavefront is 64 lanes on CDNA (gfx9xx), not 32.** This is the single biggest
  source of subtle correctness and compile issues. Audit every warp-level assumption.
- **Change the `.cu`/`.cuh`/`.cpp` sources, never the generated `.hip` files.**
  Hipify regenerates the `hip/` tree on every build; edits there are silently lost.

---

## 1. Build plumbing & hipify (fix these FIRST — they mask everything else)

### 1.1 Stale generated `hip/` trees latch and produce phantom errors
- **Symptom:** `fatal error: 'foo.cuh' file not found`, or divergent/duplicate
  header copies, or `[skipped, already hipified]` for sources but missing headers.
- **Root cause:** hipify writes a mirrored `hip/` tree next to your `cuda/` sources
  and **skips files that already exist**. If a previous build left a partial tree,
  headers that were never copied stay missing forever, and edits to sources don't
  propagate.
- **Fix:** **Delete every generated `hip/` tree before each build.** Make it part of
  your clean step:
  ```bash
  find . -type d -name hip -path '*/kernels/*' -prune -exec rm -rf {} +
  rm -rf build dist *.egg-info ~/.cache/torch_extensions/*
  ```

### 1.2 Headers not hipified/copied → "file not found"
- **Symptom:** the generated `hip/csrc/foo.hip` can't find `foo.cuh` / `foo.h`,
  even though the `.cu` next to it compiled.
- **Root cause:** torch's hipify only rewrites/copies headers it can discover via
  the **include paths you pass to the build** (`extra_include_paths` for JIT,
  include dirs for AOT). A source file's own directory is **not** automatically on
  that list. If you only add a *sibling* module's include dir, your own `.cuh`/`.h`
  never get copied into the `hip/` tree.
- **Fix:** Always put the extension's **own** `csrc`/header dir on the include path:
  ```python
  extra_include_paths = [os.path.join(PATH, "csrc"), other_module_csrc]
  ```
  This was the highest-impact bug in our port: one extension listed only a sibling's
  csrc, so none of its headers were hipified, and ~400 tests stayed "skipped."

### 1.3 Host compiler (`clang`) picks a too-new libstdc++
- **Symptom:** errors deep inside `bits/unicode.h` / `<format>` such as
  `cannot take the address of an rvalue` / `++this;` — pulled in transitively via
  `hipcub`/`rocprim`/`<format>`.
- **Root cause:** ROCm's `clang` defaults to the newest installed GCC toolchain.
  Some GCC 14 libstdc++ headers contain constructs that clang rejects but g++ tolerates.
- **Fix:** Pin clang to a known-good GCC install dir. This is a **clang-only** flag —
  put it on device flags, never on the host `g++` flags:
  ```python
  for ver in ("13", "12"):
      d = f"/usr/lib/gcc/x86_64-linux-gnu/{ver}"
      if os.path.isdir(d):
          extra_cuda_cflags.append(f"--gcc-install-dir={d}")  # NOT extra_cflags
          break
  ```
  In an AOT `setup.py` where `CXX=hipcc`, the host `.cpp` also go through hipcc/clang,
  so the flag must be on **both** the device (`hipcc`) and host (`cxx`) flag lists.

### 1.4 Linker can't find `libamdhip64`
- **Symptom:** JIT link fails with `-lamdhip64` unresolved, though compile succeeded.
- **Root cause:** torch derives `ROCM_HOME` from `hipcc` on `PATH` and emits
  `-L<venv>/lib`, but the runtime lib may live in a different site-package
  (e.g. a pip `_rocm_sdk_devel`).
- **Fix:** locate the real lib dir and add rpath+libpath:
  ```python
  extra_ldflags += [f"-L{rocm_lib}", f"-Wl,-rpath,{rocm_lib}"]
  ```

### 1.5 Vendored headers (e.g. GLM) get mangled by hipify
- **Symptom:** third-party headers emit `__device__` under `-D__HIP__` that plain
  `g++` can't parse, or hipify rewrites their includes and drops `.inl` files.
- **Fix (two options):**
  - Build host `.cpp` with `hipcc` (`CC=hipcc CXX=hipcc`) so the vendored
    `__device__` is accepted; **and/or**
  - Pass the third-party include dir as a **raw `-I`** compile flag and keep it
    **out** of `extra_include_paths`, so hipify never rewrites those headers.

### 1.6 Toolkit/JIT detection is nvcc-only
- **Symptom:** JIT extensions silently never build on HIP; features "not available."
- **Root cause:** a `cuda_available()` helper that checks for `nvcc`.
- **Fix:** treat `torch.version.hip` being set as "GPU toolkit available."

### 1.7 Environment for JIT builds
- **Symptom:** `hip/hip_runtime.h: No such file or directory` at JIT time.
- **Fix:** ensure ROCm headers are on `CPATH`/`CPLUS_INCLUDE_PATH` before importing
  (a small `source env.sh` that exports `ROCM_PATH`, `HIP_PATH`, device-lib path,
  and header includes). Also set `PYTORCH_ROCM_ARCH=gfx942` (or your arch) to avoid
  compiling for all visible archs.

---

## 2. Warp width = 64 (correctness + compile)

### 2.1 Warp-sync mask must be 64-bit
- **Symptom:**
  `static assertion failed ... 'sizeof(unsigned int) == 8': The mask must be a
  64-bit integer` (from `amd_warp_sync_functions.h`).
- **Root cause:** `__shfl_*_sync`, `__ballot_sync`, `__any_sync`, `__all_sync`,
  `__match_any_sync` take a lane mask sized to the wavefront — 64-bit on CDNA. CUDA
  code uses `unsigned int` / `0xFFFFFFFF`.
- **Fix:** introduce a portable mask type and full-mask constant, defined once in a
  shared header and used everywhere:
  ```cpp
  #if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
    using WarpMask = unsigned long long;
    constexpr WarpMask kFullWarpMask = 0xFFFFFFFFFFFFFFFFull;
  #else
    using WarpMask = unsigned int;
    constexpr WarpMask kFullWarpMask = 0xFFFFFFFFu;
  #endif
  ```
  Replace every literal `0xFFFFFFFFu` mask and `unsigned int mask` used with a
  `_sync` intrinsic. `__ballot_sync` also **returns** a 64-bit mask on HIP — store
  it in `uint64_t`, and use `__popcll`/`__ffsll` (not `__popc`/`__ffs`).

### 2.2 `warpSize` is not a compile-time 32
- **Symptom:** shared-memory sizing `[BlockThreads/32]`, `laneId = tid & 31`,
  `warpId = tid >> 5` that silently over/under-allocate or mis-reduce on 64-wide.
- **Guidance:**
  - If your algorithm genuinely wants **32-lane logical groups**, keep `kWarpSize=32`
    and `& 31` / `>> 5`, and ensure shuffle offsets stay `<= 16` so each XOR stays
    within a 32-lane half of the 64-wide wavefront. This is correct and portable.
  - If your algorithm wants a **full-warp** reduction, use a portable `kWarpSize`
    (`= warpSize` or `#if`-defined 64 on HIP) and size shared memory / loop bounds
    off it. Don't hardcode 32.
  - **Bit-packed "one bit per lane" arrays** (e.g. `visible[word] = ballot`) assume
    32-lane words — re-derive word width for 64 lanes or you corrupt neighbors.

### 2.3 Cooperative-group / mask-derived logic
- Any code that builds a mask from `__activemask()` and feeds it back into a `_sync`
  call must use the 64-bit type end-to-end.

---

## 3. half / bfloat16 intrinsics

- **Symptom:** `__hlt2_mask`, `__hgeu2_mask`, `__hle2_mask` undeclared;
  `__hmin2/__hmax2` "no matching function"; `__hip_bfloat162` ↔ `__half2` "no viable
  conversion"; `__stcs` (streaming store) undeclared.
- **Root cause:** the packed-half comparison/`*_mask` family and NVIDIA-specific
  cache-hint stores don't all exist (or have different signatures) in HIP's
  `hip_fp16.h`/`hip_bf16.h`.
- **Fix:** wrap in `#if defined(USE_ROCM)` and provide equivalents:
  - packed compares → element-wise compare of `.x`/`.y` and rebuild the mask/pair;
  - `__stcs(ptr, v)` → plain `*ptr = v;` on HIP (drop the cache hint);
  - bf16↔half → explicit `float` round-trip conversion.
- Keep the CUDA intrinsic path under `#else` so NVIDIA behavior is unchanged.

---

## 4. Vendor PTX / inline-asm helpers

- **Symptom:** inline `asm("...ptx...")` or `__nv_*` / PTX-only intrinsics fail to
  compile (no PTX on AMD GCN).
- **Fix:** guard with `#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)` and
  implement portable C++/HIP equivalents; keep the PTX under `#else`. Examples we hit:
  bitwise-select (`lop3` immLut `0xCA`), `exp2` approximations, byte-permute (`prmt`),
  packed u8↔f32 converters with round-to-nearest + saturate. **Verify numerical
  parity** against the CUDA path with a tolerance test after each such substitution.

---

## 5. `cub` → `hipcub`

- **Symptom:** `cub/block/block_scan.cuh` etc. not found; `cub::` namespace unknown.
- **Fix:**
  ```cpp
  #if defined(USE_ROCM)
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;   // or use hipcub:: directly
  #else
    #include <cub/cub.cuh>
  #endif
  ```
  Watch for API drift: some `hipFuncSetAttribute`/launch-attribute signatures differ;
  a few `cub` block primitives have different template params in `hipcub`.

---

## 6. Device-guard macros

- **Symptom:** host-only code (e.g. a throwing `TORCH_CHECK`) leaks into device
  compilation, or vice-versa.
- **Root cause:** `#ifdef __CUDA_ARCH__` does **not** fire on HIP device passes; HIP
  uses `__HIP_DEVICE_COMPILE__`.
- **Fix:** guard with both:
  ```cpp
  #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  ```
  Audit every `__CUDA_ARCH__` — some pick a valid host fallback under HIP and compile
  fine; others silently pull the wrong branch into device code.

- **`__host__ __device__` defaults:** a default constructor marked `__device__`-only
  but default-constructed in host code compiles on nvcc but fails on HIP. Mark such
  ctors `__host__ __device__`.

---

## 7. Numerical accuracy / test tolerances (the long tail)

Once it **compiles and links**, the remaining failures are usually accuracy, not
correctness bugs. Categorize them deliberately:

- **fast-math divergence:** `-use_fast_math` changes div/sqrt/transcendental
  (`sin/cos/acos/atan`) precision, and ROCm's implementations differ from NVIDIA's.
  Small element-wise diffs (1e-3…1e-2) are expected. Decide per-project whether to
  disable fast-math for accuracy-critical kernels.
- **gradcheck against float32 kernels:** `torch.autograd.gradcheck` perturbs in
  float64, but a float32-only kernel quantizes the forward, so the **numerical**
  Jacobian is noisy while the **analytical** one is clean. Cross-terms near the
  finite-difference noise floor may exceed tight `atol`. Tolerances tuned on NVIDIA
  often need loosening on ROCm because the transcendental noise floor is higher. This
  is a **test-tolerance** issue, not a kernel bug — confirm by checking that (a)
  forward correctness tests pass, (b) the analytical Jacobian is well-formed/symmetric,
  and (c) the failing gradient is w.r.t. an input that doesn't route through code you
  changed.
- **How to triage "is this my bug or platform noise?":**
  1. Does the forward/correctness test for the same op pass? → backward math is likely fine.
  2. Is the analytical result stable/symmetric across runs? → not nondeterminism.
  3. Is the diff O(1e-3–1e-2) relative on small cross-terms with matching diagonals?
     → precision, not logic.
  4. Does it fail on inputs near singularities (optical axis, division by ~0,
     iterative solvers with few iterations)? → finite-difference non-smoothness.

---

## 8. Recommended workflow

1. **Clean aggressively** (delete all `hip/` trees + caches) before every build.
2. **Fix build plumbing first** (§1) — until it links, all other errors are noise.
3. **Sweep warp-width issues** (§2) — mask types, `warpSize`, bit-packing.
4. **Sweep intrinsics** (§3–§5) — half/bf16, PTX, cub.
5. **Fix device guards** (§6).
6. **Build both AOT and JIT paths**; confirm the test harness imports the artifact
   you think it does (prebuilt `.so` vs. JIT vs. installed wheel — pin it explicitly).
7. **Run the full suite**, then **categorize** failures into: hard failures
   (build/link/crash/illegal-memory) vs. accuracy (tolerance/gradcheck). Only the
   first category blocks the port.
8. **Verify parity** for every hand-written intrinsic substitution.

---

## 9. Quick reference — portability snippets

```cpp
// Platform detection
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)   // AMD/HIP
#else                                                     // NVIDIA/CUDA
#endif

// Device-code guard (works on both)
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

// 64-bit-safe warp mask
using WarpMask = /* ull on HIP, uint on CUDA */;
__ballot_sync(kFullWarpMask, pred);   // returns 64-bit on HIP → store in uint64_t
__popcll(mask64); __ffsll(mask64);    // not __popc / __ffs

// cub
#if defined(USE_ROCM)
  #include <hipcub/hipcub.hpp>
#endif
```

```python
# JIT/AOT flag plumbing (ROCm)
if torch.version.hip:
    extra_cflags += ["-DUSE_ROCM", "-U__HIP_NO_HALF_CONVERSIONS__"]
    extra_include_paths = [own_csrc_dir, *other_dirs]   # OWN headers must be here
    extra_cuda_cflags.append(f"--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13")  # clang-only
    extra_ldflags += [f"-L{rocm_lib}", f"-Wl,-rpath,{rocm_lib}"]
```
