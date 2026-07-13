/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <ATen/DeviceGuard.h>
#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>

#ifndef USE_ROCM
#include <c10/cuda/CUDAStream.h> // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDACachingAllocator.h>
#include <cooperative_groups.h>
#include <c10/hip/CUDAGuard.h>
#else
#include <c10/hip/HIPStream.h> // at::hip::getCurrentHIPStream
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>
#include <c10/core/DeviceGuard.h>
#include <ATen/DeviceGuard.h>
#endif

namespace gsplat
{
//
// Some Macros.
//
#define CHECK_DEVICE(x)     TORCH_CHECK(x.is_cuda() || x.is_privateuseone(), #x " must be a CUDA or PrivateUse1 tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_DEVICE(x);   \
    CHECK_CONTIGUOUS(x)
// Kernels index raw dense storage; a strided (non-sparse) layout is required.
#define CHECK_DENSE(x)     TORCH_CHECK(x.layout() == c10::kStrided, #x " must be a dense tensor")

#ifndef USE_ROCM
#define GET_CURRENT_STREAM() at::cuda::getCurrentCUDAStream()
#define DEVICE_GUARD(_ten) const at::OptionalDeviceGuard device_guard(device_of(_ten));
#else
#define cub hipcub
#define GET_CURRENT_STREAM() at::hip::getCurrentHIPStream()
#define DEVICE_GUARD(_ten) const at::hip::OptionalHIPGuard device_guard(device_of(_ten));
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaSuccess hipSuccess
#endif

// Host/device qualifier for helpers shared between host (tests, host-side
// setup) and device code. Expands to nothing under a non-CUDA compiler.
#ifdef __CUDACC__
#    define GSPLAT_HOST_DEVICE __host__ __device__
#else
#    define GSPLAT_HOST_DEVICE
#endif

//
// Convenience typedefs for CUDA types
//
using vec2   = glm::vec<2, float>;
using vec3   = glm::vec<3, float>;
using vec4   = glm::vec<4, float>;
using mat2   = glm::mat<2, 2, float>;
using mat3   = glm::mat<3, 3, float>;
using mat4   = glm::mat<4, 4, float>;
using mat3x2 = glm::mat<3, 2, float>;

//
// Legacy Camera Types
//
enum CameraModelType
{
    PINHOLE = 0,
    ORTHO   = 1,
    FISHEYE = 2,
    FTHETA  = 3,
    LIDAR   = 4,
};

enum RendererConfig
{
    MIXED_BATCH    = 0,
    PARALLEL_BATCH = 1,
};

#define N_THREADS_PACKED 256

// CUDA caps grid.y (and grid.z) at 65535; only grid.x reaches 2^31 - 1. Kernels
// that map a batch (or batch-camera) dimension onto grid.y must reject launches
// that would exceed this.
constexpr uint32_t kMaxCudaGridDimY = 65535;

#define ALPHA_THRESHOLD         (1.f / 255.f)
// GAUSSIAN_EXTEND determines where the gaussian is truncated in standard deviations."
#define GAUSSIAN_EXTEND         3.33f
// MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of
// a maximal opacity Gaussian has to be rasterized twice to reach the threshold,
// without getting the transmittance too small for numerical stability of
// the backward pass.
// i.e. TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
#define MAX_ALPHA               0.99f
#define TRANSMITTANCE_THRESHOLD 1e-4f

// Floor for the antialiased compensation factor (sqrt(det_orig / det_blur)).
// Prevents compensation from reaching zero for extremely small Gaussians.
#define MIN_COMPENSATION 0.005f

// Floor for (1 - alpha) when computing 1/(1-alpha) in backward rasterization.
// Prevents gradient explosion when alpha approaches 1.0.
#define MIN_ONE_MINUS_ALPHA 1e-6f
} // namespace gsplat
