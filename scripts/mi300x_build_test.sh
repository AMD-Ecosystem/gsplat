#!/usr/bin/env bash
# Build amd_gsplat on MI300X and run the pytest suite.
set -euo pipefail

REPO="${REPO:-/home/AMD/kthatipa/aioss-gsplat}"
VENV="${VENV:-$REPO/.venv}"
ROCM_ARCHS="${ROCM_ARCHS:-gfx942}"
MAX_JOBS="${MAX_JOBS:-8}"
PYTEST_ARGS="${PYTEST_ARGS:-tests/ --tb=short -q}"

cd "$REPO"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

unset CMAKE_ROOT
export CC=hipcc CXX=hipcc ROCM_ARCHS="$ROCM_ARCHS" MAX_JOBS="$MAX_JOBS"

echo "==> Clean HIPify cache and build artifacts"
rm -rf build/ gsplat/hip/ dist/

echo "==> build_ext"
python setup.py build_ext 2>&1 | tee /tmp/gsplat_mi300x_build.log

echo "==> bdist_wheel (--skip-build)"
python setup.py bdist_wheel --skip-build

echo "==> pip install wheel"
pip install dist/amd_gsplat*.whl --force-reinstall --no-deps

echo "==> Copy csrc.so into repo tree for dev imports"
CSRC="$(python - <<'PY'
import gsplat.csrc as csrc, pathlib
print(pathlib.Path(csrc.__file__).resolve())
PY
)"
cp -f "$CSRC" "$REPO/gsplat/csrc.so"

echo "==> Sanity check"
python - <<'PY'
import torch, gsplat
print("torch", torch.__version__, "hip", torch.version.hip, "cuda", torch.cuda.is_available())
print("has_3dgs", gsplat.has_3dgs(), "has_2dgs", gsplat.has_2dgs())
import torch.ops.gsplat as ops
for name in ("rasterize_to_pixels_3dgs", "projection_ewa_simple", "rasterize_to_pixels_2dgs"):
    print(name, hasattr(ops, name))
PY

echo "==> pytest"
pytest $PYTEST_ARGS 2>&1 | tee /tmp/gsplat_pytest.log
echo "EXIT:${PIPESTATUS[0]}" >> /tmp/gsplat_pytest.log
