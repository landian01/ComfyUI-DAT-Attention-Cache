// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 landian01
#include <climits>
#include <cstdint>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

namespace {

__device__ __forceinline__ int region(int coordinate, int length, int window, int shift) {
    if (shift == 0) return 2;
    return coordinate >= length - shift ? 2 : (coordinate >= length - window ? 1 : 0);
}

template <typename scalar_t, int Tokens>
__global__ void dat_mask_kernel(
    scalar_t* first, scalar_t* second, int height, int width,
    int split_h, int split_w, int shift_h, int shift_w, int dynamic_tokens) {
    const bool swapped = blockIdx.y != 0;
    const int wh = swapped ? split_w : split_h;
    const int ww = swapped ? split_h : split_w;
    const int sh = swapped ? shift_w : shift_h;
    const int sw = swapped ? shift_h : shift_w;
    const int tokens = Tokens == 0 ? dynamic_tokens : Tokens;
    const int columns = width / ww;
    const int window_y = blockIdx.x / columns;
    const int window_x = blockIdx.x % columns;
    const int origin_y = window_y * wh;
    const int origin_x = window_x * ww;
    const bool internal = window_y < height / wh - 1 && window_x < columns - 1;
    scalar_t* output = (swapped ? second : first) + int64_t(blockIdx.x) * tokens * tokens;
    for (int index = threadIdx.x; index < tokens * tokens; index += blockDim.x) {
        bool equal = true;
        if (!internal) {
            const int i = index / tokens;
            const int j = index % tokens;
            const int yi = origin_y + i / ww;
            const int yj = origin_y + j / ww;
            const int xi = origin_x + i % ww;
            const int xj = origin_x + j % ww;
            equal = region(yi, height, wh, sh) == region(yj, height, wh, sh)
                && region(xi, width, ww, sw) == region(xj, width, ww, sw);
        }
        output[index] = static_cast<scalar_t>(equal ? 0.0f : -100.0f);
    }
}

std::tuple<at::Tensor, at::Tensor> make_masks(
    int64_t height, int64_t width, int64_t split_h, int64_t split_w,
    int64_t shift_h, int64_t shift_w, c10::Device device, c10::ScalarType dtype) {
    TORCH_CHECK(device.is_cuda(), "DAT mask requires a CUDA device");
    TORCH_CHECK(height > 0 && width > 0 && split_h > 0 && split_w > 0,
                "DAT mask dimensions must be positive");
    TORCH_CHECK(height <= INT_MAX && width <= INT_MAX
                    && split_h <= INT_MAX && split_w <= INT_MAX,
                "DAT mask dimensions exceed kernel limits");
    TORCH_CHECK(height % split_h == 0 && height % split_w == 0
                    && width % split_h == 0 && width % split_w == 0,
                "DAT mask dimensions must be divisible by both window dimensions");
    TORCH_CHECK(shift_h >= 0 && shift_h <= split_h && shift_w >= 0 && shift_w <= split_w,
                "DAT mask shifts must be within the window");
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
                "DAT mask supports float32, float16 and bfloat16");
    const int64_t tokens = split_h * split_w;
    const int64_t windows = (height / split_h) * (width / split_w);
    TORCH_CHECK(tokens <= 1024 && windows <= INT_MAX,
                "DAT mask dimensions exceed kernel limits");
    const c10::cuda::CUDAGuard guard(device);
    auto options = at::TensorOptions().device(device).dtype(dtype);
    auto first = at::empty({windows, tokens, tokens}, options);
    auto second = at::empty({windows, tokens, tokens}, options);
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    dim3 grid(static_cast<unsigned int>(windows), 2);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dtype, "dat_mask", [&] {
        if (tokens == 128) {
            dat_mask_kernel<scalar_t, 128><<<grid, 256, 0, stream>>>(
                first.data_ptr<scalar_t>(), second.data_ptr<scalar_t>(),
                height, width, split_h, split_w, shift_h, shift_w, tokens);
        } else {
            dat_mask_kernel<scalar_t, 0><<<grid, 256, 0, stream>>>(
                first.data_ptr<scalar_t>(), second.data_ptr<scalar_t>(),
                height, width, split_h, split_w, shift_h, shift_w, tokens);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {first, second};
}

}  // namespace

TORCH_LIBRARY(dat_attention_cache_native, m) {
    m.def("make_masks(int height, int width, int split_h, int split_w, "
          "int shift_h, int shift_w, Device device, ScalarType dtype) -> (Tensor, Tensor)",
          &make_masks);
}
