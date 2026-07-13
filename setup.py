import glob
import os
import os.path as osp
import pathlib
import platform
import sys
import re
import site
import subprocess

from setuptools import find_packages, setup

IS_ROCM = True


def _find_rocm_sdk_in_site_packages():
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        if not sp:
            continue
        sdk_devel = osp.join(sp, "_rocm_sdk_devel")
        if osp.isdir(osp.join(sdk_devel, "include", "hip")):
            return sdk_devel
        sdk_core = osp.join(sp, "_rocm_sdk_core")
        if osp.isdir(osp.join(sdk_core, "include", "hip")):
            return sdk_core
    return None


def _bootstrap_rocm_env():
    """Point ROCm tooling at pip SDK (devel preferred) before torch import."""
    rocm_home = os.environ.get("ROCM_HOME")
    if not rocm_home or not osp.isdir(osp.join(rocm_home, "include", "hip")):
        rocm_home = None
        if osp.isdir(osp.join("/opt/rocm", "include", "hip")):
            rocm_home = "/opt/rocm"
        else:
            rocm_home = _find_rocm_sdk_in_site_packages() or "/opt/rocm"

    rocm_home = osp.abspath(rocm_home)
    rocm_bin = osp.join(rocm_home, "bin")
    rocm_llvm_bin = osp.join(rocm_home, "lib", "llvm", "bin")
    os.environ["ROCM_HOME"] = rocm_home
    os.environ["ROCM_PATH"] = rocm_home
    os.environ["PATH"] = os.pathsep.join(
        [rocm_bin, rocm_llvm_bin, os.environ.get("PATH", "")]
    )


if IS_ROCM:
    _bootstrap_rocm_env()

import torch

__version__ = None
exec(open("gsplat/version.py", "r").read())


def get_rocm_sdk_root():
    """Resolve ROCm SDK root: ROCM_HOME, devel, core, or /opt/rocm."""
    rocm_home = os.environ.get("ROCM_HOME")
    if rocm_home and osp.isdir(osp.join(rocm_home, "include", "hip")):
        return osp.abspath(rocm_home)
    if osp.isdir(osp.join("/opt/rocm", "include", "hip")):
        return "/opt/rocm"
    found = _find_rocm_sdk_in_site_packages()
    if found:
        return found
    return "/opt/rocm"


def get_rocm_include_dirs(rocm_home):
    """Collect ROCm-related include directories (HIP, thrust, rocprim, hipcub)."""
    repo_root = osp.dirname(osp.abspath(__file__))
    sdk_include = osp.join(rocm_home, "include")
    include_dirs = [
        sdk_include,
        osp.expanduser("~/.local/include"),
    ]

    extra_includes = os.environ.get("ROCM_EXTRA_INCLUDES", "")
    if extra_includes:
        include_dirs.extend(
            p.strip() for p in extra_includes.split(os.pathsep) if p.strip()
        )

    sdk_has_thrust = osp.isfile(osp.join(sdk_include, "thrust", "complex.h"))
    sdk_has_rocprim = osp.isfile(
        osp.join(sdk_include, "rocprim", "type_traits.hpp")
    )
    sdk_has_hipcub = osp.isfile(osp.join(sdk_include, "hipcub", "hipcub.hpp"))
    if sdk_has_thrust and sdk_has_rocprim and sdk_has_hipcub:
        # Prefer the pip SDK tree as a single consistent header set.
        seen = set()
        unique_dirs = []
        for d in include_dirs:
            if d not in seen and osp.isdir(d):
                seen.add(d)
                unique_dirs.append(d)
        return unique_dirs

    thrust_candidates = [
        os.environ.get("ROCM_THRUST_INCLUDE"),
        osp.join(repo_root, ".rocm_deps", "rocthrust"),
        osp.expanduser("~/.local/rocthrust"),
        sdk_include,
        "/opt/rocm/include",
    ]
    rocprim_candidates = [
        osp.join(repo_root, ".rocm_deps", "rocprim", "rocprim", "include"),
        sdk_include,
        "/opt/rocm/include",
    ]
    hipcub_candidates = [
        osp.join(repo_root, ".rocm_deps", "hipcub", "hipcub", "include"),
        sdk_include,
        "/opt/rocm/include",
    ]

    for path in thrust_candidates:
        if path and osp.isfile(osp.join(path, "thrust", "complex.h")):
            include_dirs.append(path)
            break

    for path in rocprim_candidates:
        if path and osp.isfile(osp.join(path, "rocprim", "type_traits.hpp")):
            include_dirs.append(path)
            break

    for path in hipcub_candidates:
        if path and osp.isfile(osp.join(path, "hipcub", "hipcub.hpp")):
            include_dirs.append(path)
            break

    # Deduplicate while preserving order.
    seen = set()
    unique_dirs = []
    for d in include_dirs:
        if d not in seen and osp.isdir(d):
            seen.add(d)
            unique_dirs.append(d)
    return unique_dirs


def get_rocm_archs():
    """
    Return target GPU architectures for HIP offload.

    Uses ROCM_ARCHS env (comma-separated), else rocminfo if available,
    else defaults to gfx942 and gfx950 for cross-compile without a GPU.
    """
    env_archs = os.environ.get("ROCM_ARCHS", "").strip()
    if env_archs:
        archs = [a.strip() for a in env_archs.split(",") if a.strip()]
        print(f"Using ROCM_ARCHS from environment: {archs}")
        return archs

    if ENABLE_TEST_COVERAGE:
        return ["gfx942"]

    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"Name:\s+(gfx[0-9a-z]+)", result.stdout)
        if not match:
            match = re.search(r"\b(gfx[0-9a-z]+)\b", result.stdout)
        if match:
            gfx_code = match.group(1)
            print(f"Detected ROCm GPU architecture from rocminfo: {gfx_code}")
            return [gfx_code]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    default_archs = ["gfx942", "gfx950"]
    print(
        "rocminfo unavailable or no GPU detected; "
        f"using default architectures: {default_archs}"
    )
    return default_archs

def is_git_repo(folder_path):
    """
    Checks if a folder is a Git repository by running 'git rev-parse --git-dir'.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        bool: True if it is a Git repository, False otherwise.
    """
    # First, check if the folder path is a valid directory
    if not os.path.isdir(folder_path):
        return False

    try:
        # Run the git command
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=folder_path,
            capture_output=True,
            text=True
        )

        # The command returns an exit code of 0 if it's a repo
        return result.returncode == 0

    except FileNotFoundError:
        # This exception is raised if 'git' is not in the system's PATH
        print("Error: Git is not installed or not in your system's PATH.")
        return False
    except Exception as e:
        # Handle other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return False

def get_git_rev(folder_path):
    """
    Checks if a folder is a Git repository and returns the latest commit SHA
    of the main branch using Git command-line tools.

    Args:
        folder_path (str): The path to the folder to check.

    Returns:
        str: The SHA of the latest commit on the main branch, or None if
             it's not a Git repository or the main branch doesn't exist.
    """
    try:
        # Use subprocess to run 'git rev-parse main' to get the commit hash
        # 'rev-parse' is a low-level command used to translate a human-readable
        # name into an SHA-1.
        command = ['git', 'rev-parse', '--short' ,"HEAD"]

        # Run the command in the specified folder
        result = subprocess.run(
            command,
            cwd=folder_path,
            capture_output=True,
            text=True,
            check=True
        )

        # The output is the commit hash
        commit_sha = result.stdout.strip()
        return commit_sha

    except subprocess.CalledProcessError as e:
        # This error is raised if the command fails, which happens if 'main'
        # branch doesn't exist
        print(f"Error: The #{branch_name} branch does not exist or another error occurred: {e}")
        return ""
    except FileNotFoundError:
        print("Error: Git is not installed or not in your system's PATH.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""
    return ""

if is_git_repo:
    git_rev = get_git_rev(os.getcwd())
    __version__ += f"+{git_rev}"

print(f"VERSION = {__version__}")

URL = "https://github.com/rocm/gsplat"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS =  os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"

ENABLE_TEST_COVERAGE = os.getenv("ENABLE_TEST_COVERAGE", "0") == "1"

MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"
    print(f"Setting MAX_JOBS to {os.environ['MAX_JOBS']}")

def get_ext():
    from torch.utils.cpp_extension import BuildExtension
    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    if IS_ROCM:
        from torch.utils.cpp_extension import CUDAExtension
        print("ROCM detected, compiling with HIP support...")
        rocm_home = get_rocm_sdk_root()
        rocm_bin = osp.join(rocm_home, "bin")
        rocm_llvm_bin = osp.join(rocm_home, "lib", "llvm", "bin")
        os.environ["ROCM_HOME"] = rocm_home
        os.environ["ROCM_PATH"] = rocm_home
        os.environ["PATH"] = os.pathsep.join(
            [rocm_bin, rocm_llvm_bin, os.environ.get("PATH", "")]
        )
        rocm_lib = osp.join(rocm_home, "lib")
        gpu_archs = get_rocm_archs()
        print(f"ROCm SDK root: {rocm_home}")
        print(f"GPU architectures: {gpu_archs}")

        extensions_dir = osp.join("gsplat", "cuda")
        sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(
            osp.join(extensions_dir, "csrc", "*.cpp")
        )
        sources += [osp.join(extensions_dir, "ext.cpp")]

        undef_macros = []
        define_macros = []

        cxx_flags = [
            "-D__HIP_PLATFORM_AMD__",
            "-Wno-sign-compare",
            "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
            "-DUSE_ROCM",
            "-DAT_PARALLEL_OPENMP",
            "-fopenmp",
        ]
        if WITH_SYMBOLS:
            cxx_flags += ["-g", "-O0"]
        else:
            cxx_flags += ["-O3", "-Wno-attributes", "-Wno-switch", "-Wno-comment"]
        extra_compile_args = {"cxx": cxx_flags}

        extra_link_args = [] if WITH_SYMBOLS else ["-s"]

        hipcc_flags = [
            "-D__HIP_PLATFORM_AMD__",
            "-D__HIP_PLATFORM_HCC__=1",
            "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
            "-DUSE_ROCM",
            f"--rocm-path={rocm_home}",
        ]
        for arch in gpu_archs:
            hipcc_flags.append(f"--offload-arch={arch}")
        if WITH_SYMBOLS:
            hipcc_flags += ["-g", "-ggdb", "-O0"]
        else:
            hipcc_flags += ["-O3"]
        if LINE_INFO:
            hipcc_flags += ["-gline-tables-only"]
        if torch.version.hip:
            define_macros += [("USE_ROCM", "1")]
            undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
        if ENABLE_TEST_COVERAGE:
            extra_compile_args["cxx"] += [
                "-fprofile-instr-generate",
                "-fcoverage-mapping",
                "-Qunused-arguments",
                "--gcc-toolchain=/usr",
            ]
            hipcc_flags += ["-fprofile-instr-generate", "-fcoverage-mapping"]
            extra_link_args += ["-fprofile-instr-generate"]

        extra_compile_args["nvcc"] = hipcc_flags
        current_dir = pathlib.Path(__file__).parent.resolve()

        include_dirs = get_rocm_include_dirs(rocm_home)
        include_dirs.insert(
            0, osp.join(current_dir, "gsplat", "cuda", "include")
        )
        print(f"ROCm include dirs: {include_dirs}")
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            include_dirs.append(osp.join(conda_prefix, "include"))

        library_dirs = [rocm_lib]
        if conda_prefix:
            library_dirs.append(osp.join(conda_prefix, "lib"))

        extension = CUDAExtension(
            "gsplat.csrc",
            sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        return [extension]
    else:

        from torch.__config__ import parallel_info
        from torch.utils.cpp_extension import CUDAExtension

        extensions_dir = osp.join("gsplat", "cuda")
        sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(
            osp.join(extensions_dir, "csrc", "*.cpp")
        )
        sources += [osp.join(extensions_dir, "ext.cpp")]

        undef_macros = []
        define_macros = []

        extra_compile_args = {"cxx": ["-O3"]}
        if not os.name == "nt":  # Not on Windows:
            extra_compile_args["cxx"] += ["-Wno-sign-compare"]
        extra_link_args = [] if WITH_SYMBOLS else ["-s"]

        info = parallel_info()
        if (
            "backend: OpenMP" in info
            and "OpenMP not found" not in info
            and sys.platform != "darwin"
        ):
            extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
            if sys.platform == "win32":
                extra_compile_args["cxx"] += ["/openmp"]
            else:
                extra_compile_args["cxx"] += ["-fopenmp"]
        else:
            print("Compiling without OpenMP...")

        # Compile for mac arm64
        if sys.platform == "darwin" and platform.machine() == "arm64":
            extra_compile_args["cxx"] += ["-arch", "arm64"]
            extra_link_args += ["-arch", "arm64"]

        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
        nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17"]
        if LINE_INFO:
            nvcc_flags += ["-lineinfo"]
        if torch.version.hip:
            # USE_ROCM was added to later versions of PyTorch.
            # Define here to support older PyTorch versions as well:
            define_macros += [("USE_ROCM", None)]
            undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
        else:
            nvcc_flags += ["--expt-relaxed-constexpr"]

    # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses
    nvcc_flags += ["-diag-suppress", "20012,186"]
    extra_compile_args["nvcc"] = nvcc_flags
    if sys.platform == "win32":
        extra_compile_args["nvcc"] += [
            "-DWIN32_LEAN_AND_MEAN",
            "-allow-unsupported-compiler",
        ]

        current_dir = pathlib.Path(__file__).parent.resolve()
        glm_path = osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm")
        include_dirs = [glm_path, osp.join(current_dir, "gsplat", "cuda", "include")]

        extension = CUDAExtension(
            "gsplat.csrc",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        return [extension]

import torch.utils.cpp_extension as ce

def fixed_get_compiler_abi_compatibility_and_version(compiler):
    try:
        return ce.original_get_compiler_abi_compatibility_and_version(compiler)
    except ValueError:
        # Fallback for clang++ "17.0git"
        return ("gcc", (17, 0))

if not hasattr(ce, "original_get_compiler_abi_compatibility_and_version"):
    ce.original_get_compiler_abi_compatibility_and_version = ce.get_compiler_abi_compatibility_and_version
    ce.get_compiler_abi_compatibility_and_version = fixed_get_compiler_abi_compatibility_and_version


setup(
    name="amd_gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
	author="AMD Corporation",
    license="Apache 2.0",
    python_requires=">=3.7",
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "build",
            "twine",
        ],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": get_ext()},
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)

if need_to_unset_max_jobs:
    print("Unsetting MAX_JOBS")
    os.environ.pop("MAX_JOBS")

