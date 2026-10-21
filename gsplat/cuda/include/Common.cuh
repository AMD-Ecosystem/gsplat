#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>

#ifndef USE_ROCM
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <c10/hip/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>

#else
#include <hip/hip_cooperative_groups.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/hip/Atomic.cuh>
#include <hipcub/hipcub.hpp>
#include <hipcub/block/block_reduce.hpp>
#include <rocprim/warp/warp_reduce.hpp>
// Hardware wavefront size of the device compile target: 64 on CDNA (gfx9xx),
// 32 on RDNA (gfx10/11/12). Evaluates to 32 in the host compilation pass.
#ifndef GSPLAT_WARP_SIZE
#define GSPLAT_WARP_SIZE (::rocprim::arch::wavefront::min_size())
#endif
#endif

namespace gsplat {

#ifndef USE_ROCM
// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...)                                                 \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)
#else
#define cub hipcub
#define CUB_WRAPPER(func, ...)                                                 \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        auto res = func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::hip::HIPCachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        res = func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);  \
	assert(res == hipSuccess);                                            \
    } while (false)
#endif

#ifdef USE_ROCM
inline int device_wavefront_size(int device) {
    int width = 0;
    TORCH_CHECK(
        hipDeviceGetAttribute(&width, hipDeviceAttributeWarpSize, device) ==
            hipSuccess,
        "hipDeviceGetAttribute(hipDeviceAttributeWarpSize) failed"
    );
    return width;
}

// The rasterization backward kernels reduce across a whole wavefront, so a tile must hold
// at least one. With fewer threads the GPU raises HSA_STATUS_ERROR_EXCEPTION 0x1016, which
// aborts the process rather than returning an error.
inline void check_tile_size(uint32_t tile_size, int wavefront_size) {
    TORCH_CHECK(
        tile_size * tile_size >= static_cast<uint32_t>(wavefront_size),
        "tile_size=",
        tile_size,
        " gives ",
        tile_size * tile_size,
        " threads per block, fewer than this GPU's wavefront of ",
        wavefront_size,
        " threads. The rasterization backward pass reduces across a whole wavefront. "
        "Use tile_size 8 or larger."
    );
}
#endif
} // namespace gsplat
