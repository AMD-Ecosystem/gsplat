#!/bin/bash
set -euo pipefail

# AMD ROCm nightly (whl-next). Latest checked 2026-11-05: 10.3.0a20261105.
#
# Why 10.3.0a (and the HSA_DISABLE_GFX12_STRICT=1 flag):
#   * 10.2.0a shipped NO gfx1250v0 Tensile library, so every hipBLASLt GEMM on
#     gfx1250 A0 crashed (HIPBLAS_STATUS_INVALID_VALUE). 10.3.0a fixes that --
#     the gfx1250v0 TensileLibrary_lazy_gfx1250.dat master index now ships.
#   * BUT 10.3.0a enumerates the A0 device as "gfx1250-strict" by default, and
#     the shipped kernel code objects are not built for that strict ISA, so GPU
#     kernel loads fail with hipErrorInvalidKernelFile ("invalid kernel file").
#   * Setting HSA_DISABLE_GFX12_STRICT=1 makes the device enumerate as plain
#     "gfx1250"; GEMMs then run through hipBLASLt with NO TORCH_BLAS_PREFER_
#     HIPBLASLT=0 workaround needed. This flag is a RUNTIME setting -- export it
#     in every shell that runs torch/gsplat (see RUNTIME section at the bottom).
#   * Caveat: with strict disabled, hipBLASLt uses the generic gfx1250 library
#     rather than the A0-specific gfx1250v0 one, so A0 numerics are provisional.
INDEX_URL="https://nightly.repo.amd.com/rocm/whl-next/"
VERSION_TAG="10.3.0a20261105"
GFX_ARCH="gfx1250"

TORCH_VERSION="2.14.0"
TORCHVISION_VERSION="0.29.0"

# ---------------------------------------------------------------------------
# Safety guards (learned the hard way):
#   * A Python env MUST be active (venv OR conda), otherwise pip installs into
#     system/user site.
#   * That env MUST live OUTSIDE the gsplat source tree. If the ROCm SDK sits
#     under the build dir, torch's hipify re-hipifies the SDK headers in place
#     (generating *_hip.h twins) and the HIP build dies with thrust/hipcub
#     "redefinition" errors. Keep the env out of the repo.
# ---------------------------------------------------------------------------
ACTIVE_ENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-}}"
if [[ -z "${ACTIVE_ENV}" ]]; then
    echo "ERROR: no Python env is active. Create one OUTSIDE the repo first, e.g.:" >&2
    echo "  venv:  python3.12 -m venv ~/venvs/gsplat-eco && source ~/venvs/gsplat-eco/bin/activate" >&2
    echo "  conda: conda create -n gsplat -c conda-forge python=3.12 && conda activate gsplat" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_REAL="$(cd "${ACTIVE_ENV}" && pwd -P)"
if [[ "${ENV_REAL}" == "${REPO_ROOT}"/* ]]; then
    echo "ERROR: the active Python env is inside the repo:" >&2
    echo "         env:  ${ENV_REAL}" >&2
    echo "         repo: ${REPO_ROOT}" >&2
    echo "This causes hipify to corrupt the ROCm SDK on build. Recreate it outside" >&2
    echo "the repo and re-run." >&2
    exit 1
fi

echo "======================================================="
echo "Installing ROCm ${VERSION_TAG} for ${GFX_ARCH}"
echo "  torch==${TORCH_VERSION}+rocm${VERSION_TAG}"
echo "  torchvision==${TORCHVISION_VERSION}+rocm${VERSION_TAG}"
echo "  python=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "  env=${ENV_REAL}"
echo "======================================================="

echo "======================================================="
echo "1. Installing ROCm Multi-Arch SDK"
echo "======================================================="
python -m pip install \
  -i "${INDEX_URL}" --pre \
  "rocm[libraries,devel]==${VERSION_TAG}" \
  "rocm-sdk-device-${GFX_ARCH}==${VERSION_TAG}"

echo "======================================================="
echo "2. Installing PyTorch & Ecosystem for ROCm"
echo "======================================================="
python -m pip install \
  -i "${INDEX_URL}" --pre \
  "torch==${TORCH_VERSION}+rocm${VERSION_TAG}" \
  "amd-torch-device-${GFX_ARCH}==${TORCH_VERSION}+rocm${VERSION_TAG}" \
  "torchvision==${TORCHVISION_VERSION}+rocm${VERSION_TAG}" \
  "amd-torchvision-device-${GFX_ARCH}==${TORCHVISION_VERSION}+rocm${VERSION_TAG}"

echo "======================================================="
echo "Done. Quick check (incl. a hipBLASLt GEMM smoke test):"
echo "======================================================="
# A0 requires this flag at RUNTIME; export it before torch initializes so the
# device enumerates as gfx1250 and GEMMs do not hit hipErrorInvalidKernelFile.
export HSA_DISABLE_GFX12_STRICT=1
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"hip={getattr(torch.version, 'hip', None)}")
try:
    import torchvision
    print(f"torchvision={torchvision.__version__}")
except Exception as exc:
    print(f"torchvision=MISSING ({exc})")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device0={torch.cuda.get_device_name(0)}")
    print(f"arch={torch.cuda.get_device_properties(0).gcnArchName}")
    print(f"device_count={torch.cuda.device_count()}")
    # hipBLASLt GEMM smoke test (default hipBLASLt path, no TORCH_BLAS workaround)
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print(f"gemm_smoke_test=OK (checksum={float(c.sum()):.1f})")
PY

echo "======================================================="
echo "RUNTIME: export this in EVERY shell that runs torch/gsplat on gfx1250 A0:"
echo "  export HSA_DISABLE_GFX12_STRICT=1"
echo "(No TORCH_BLAS_PREFER_HIPBLASLT=0 needed on 10.3.0a.)"
echo "======================================================="
