# gsplat for ROCm

gsplat is an open-source library for GPU-accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

This repo is the HIP port of the gsplat library, enabled and optimized for ROCm on AMD Instinct processors. It tracks upstream [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) (currently v1.5.3) while preserving AMD-specific defaults and build paths.

## System Requirements

- OS: Ubuntu 22.04 and above
- ROCm: 6.4.1 and above
- GPU platforms: MI300X
- PyTorch: 2.6 and above
- Python: 3.12 and above

## News

### v1.6.0

Changes on `main` since the [v1.5.3](https://github.com/nerfstudio-project/gsplat/releases/tag/v1.5.3) tag (not yet on PyPI).

- [Aug 2026] **Spherical-harmonics updates** -- split diffuse and view-dependent operators, fused view-direction computation, and optimized backward kernels.
- [Jul 2026] **Sparse 3DGS rasterization** -- active-tile rendering with forward/backward support and sparse Gaussian ID, count, and top-contributor queries.
- [Jul 2026] **Multi-GPU dense 3DGS** -- distributed support for the dense rendering path.
- [Jul 2026] **PPISP across densification strategies** -- post-processing now works with both the default and MCMC strategies.
- [Jul 2026] **Additional performance and code improvements**:
  - **Faster 3DGS and 3DGUT kernels** -- optimized dense rasterization, spherical-harmonics feature assembly, and projection across sensor models.
  - **Orthographic-camera support** for 3DGUT.
  - **Fused SE(3) operations** for pose composition and trajectory interpolation.
  - **More configurable MCMC operations** for perturbation and relocation.
  - **PyTorch 2.7+ is now required**, with improved CUDA 12.8 and 13.2 build compatibility.
- [Jun 2026] **G-SHARP** -- dynamic surgical-scene reconstruction for Gaussian splatting.
- [Jun 2026] **New camera & sensor support** in the sensors library: pinhole, FTheta, fisheye, and LiDAR models.
- [Jun 2026] **Faster rasterization** -- performance improvements across the 3DGS and 3DGUT rasterization paths.
- [Jun 2026] **Gaussian ID rasterization** -- rasterize Gaussian IDs, counts, and top contributors per pixel.
- [Jun 2026] **Various performance and code improvements**: spherical-harmonics improvements, profiling/tracing tools, robustness fixes, stronger test coverage, CUDA 13 and NumPy 2 support, simplified package layout.
- [May 2026] **Inference Rendering (HiGS)** -- experimental inference-only rendering under `gsplat.experimental`.
- [May 2026] Native **MCMC perturb** (`inject_noise`) for MCMC-style Gaussian optimization.
- [Apr 2026] **AccuTile** conservative ellipse-based tile–Gaussian intersection test.
- [Apr 2026] **NCore v4** capture support.
- [Mar 2026] **LiDAR** rasterization for 3D Gaussian splatting.
- [Mar 2026] **TorchScript-oriented** deployment for camera models and distortions.
- [Mar 2026] **3DGUT** extensions: external distortion, per-ray inputs, ray-normal outputs.
- [Jan 2026] [PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/) integrated as an alternative to bilateral grid.

## Installation

Install [PyTorch for ROCm](https://pytorch.org/get-started/locally/) first. The easiest way is the official ROCm docker image:

ROCm 6.4:

```bash
docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true --shm-size=128GB --network=host --device=/dev/kfd --device=/dev/dri --group-add video -it -v $HOME:$HOME --name rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
```

ROCm 7.0:

```bash
docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true --shm-size=128GB --network=host --device=/dev/kfd --device=/dev/dri --group-add video -it -v $HOME:$HOME --name rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0
```

Install gsplat from the AMD-hosted PyPI repository:

```bash
pip install gsplat --extra-index-url=https://pypi.amd.com/simple
```

Verify installation:

```bash
pip show gsplat
```

## Examples

Get the examples folder from the GitHub repo:

```bash
git clone --no-checkout https://github.com/rocm/gsplat.git && cd gsplat && git sparse-checkout init --cone
git sparse-checkout add examples && git checkout main
```

Install dependencies and download the test dataset:

```bash
cd examples
./install_dependencies.sh
python datasets/download_dataset.py
```

AMD-specific walkthroughs:

- [Fit a Single Image](https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1128163102/Fit+a+single+image)
- [Fit a COLMAP Capture](https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1128134447/Fit+a+COLMAP+Capture)
- [Render a Large Scene](https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1128302047/Render+a+Large+Scene)

Upstream examples (require `pip install -r examples/requirements.txt --no-build-isolation`):

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)
- [Train on an NCore v4 capture.](https://docs.gsplat.studio/main/examples/ncore.html)

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat's efficient GPU implementation, training takes up to **4x less GPU memory** with up to **15% less time** than the official implementation. Full AMD report: [Evaluation Results on MI300](https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1115965044/Evaluation+Results+on+MI300). Upstream CUDA report: [evaluation results](https://docs.gsplat.studio/main/tests/eval.html).

```bash
python -m pip install -e .
cd examples
python -m pip install -r requirements.txt
python datasets/download_dataset.py
bash benchmarks/basic.sh
```

## Inference Rendering

gsplat includes an experimental inference-only rendering path based on HiGS (Hierarchically Tiled Gaussian Splatting) in `gsplat.experimental`, designed for low-latency rendering of pre-trained Gaussian scenes. See the [HiGS project page](https://research.nvidia.com/labs/sil/projects/higs/) and `examples/benchmarks/gaussian_render_inference_scene/`.

```python
from gsplat.experimental import render_scene, GaussianInferenceScene
```

## Building from Source

For ROCm/HIP builds, see [docs/DevSetup.md](docs/DevSetup.md).

For general development workflow (formatting, tests, docs), see [docs/DEV.md](docs/DEV.md).

## Contributing

We welcome contributions of any kind. See [Contributing.md](docs/Contributing.md) for details.

## Development and Contribution

This project is developed by contributors from the following institutes (unordered):

- UC Berkeley
- NVIDIA
- ShanghaiTech University
- Amazon
- Meta
- IIIT
- LumaAI
- SpectacularAI
- Aalto University
- CMU

We also have a white paper about the project with benchmarking and mathematical supplement, available [on arXiv](https://arxiv.org/abs/2409.06765). If you find this library useful in your projects or papers, please consider citing:

```
@article{ye2025gsplat,
  title={gsplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={34},
  pages={1--17},
  year={2025}
}
```
