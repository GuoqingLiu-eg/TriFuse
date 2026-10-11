// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
#pragma once

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/version.h>

#include <stdexcept>
#include <utility>

namespace executorch::backends::mlx {

// Pinned 0.32.2/0.32.3 Metal softmax_single_row<half,float> with E=64:
// 32 lanes, four consecutive values per lane, two SIMD reduction stages.
// Ranking must happen AFTER the explicit Half probability rounding. The
// pinned Metal ArgPartition uses stable ascending sort of -probabilities.
inline constexpr const char* kOLMoERouterSource = R"TRIFUSE_ROUTE(
#pragma clang fp reassociate(off) contract(off)
const uint lane = thread_index_in_simdgroup;
threadgroup float local_max[32];
threadgroup float local_normalizer[32];
threadgroup half probabilities[64];
float ld[4];
for (uint i = 0; i < 4; ++i) {
    const uint index = lane * 4 + i;
    ld[i] = index < 64 ? float(router_logits[index]) : -metal::numeric_limits<float>::infinity();
}
local_max[lane] = -metal::numeric_limits<float>::infinity();
local_normalizer[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);

float maxval = -metal::numeric_limits<float>::max();
for (uint i = 0; i < 4; ++i) {
    maxval = maxval < ld[i] ? ld[i] : maxval;
}
maxval = metal::simd_max(maxval);
if (lane == 0) local_max[0] = maxval;
threadgroup_barrier(mem_flags::mem_threadgroup);
maxval = metal::simd_max(local_max[lane]);
if (lane == 0) local_max[0] = maxval;
threadgroup_barrier(mem_flags::mem_threadgroup);
maxval = local_max[0];

float normalizer = 0.0f;
for (uint i = 0; i < 4; ++i) {
    const float exp_x = metal::fast::exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
}
normalizer = metal::simd_sum(normalizer);
if (lane == 0) local_normalizer[0] = normalizer;
threadgroup_barrier(mem_flags::mem_threadgroup);
normalizer = metal::simd_sum(local_normalizer[lane]);
if (lane == 0) local_normalizer[0] = normalizer;
threadgroup_barrier(mem_flags::mem_threadgroup);
normalizer = 1.0f / local_normalizer[0];
for (uint i = 0; i < 4; ++i) {
    const uint index = lane * 4 + i;
    if (index < 64) probabilities[index] = half(ld[i] * normalizer);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Stable descending Half probability, NaN last, with increasing expert ID
// for ties. Every real expert receives a unique rank, including zero/NaN.
for (uint i = lane; i < 64; i += 32) {
    const half value = probabilities[i];
    const bool value_nan = metal::isnan(value);
    uint rank = 0;
    for (uint j = 0; j < 64; ++j) {
        const half other = probabilities[j];
        const bool other_nan = metal::isnan(other);
        const bool before = !other_nan && (value_nan || other > value);
        const bool equal = (other_nan && value_nan) ||
                           (!other_nan && !value_nan && other == value);
        rank += uint(before || (equal && j < i));
    }
    if (rank < 8) {
        indices[rank] = i;
        gates[rank] = value;
    }
}
)TRIFUSE_ROUTE";

inline std::pair<::mlx::core::array, ::mlx::core::array> olmoe_route(
    const ::mlx::core::array& logits,
    int tokens,
    int experts,
    int experts_per_token,
    const ::mlx::core::Stream& stream) {
  using namespace ::mlx::core;
  const int T = tokens, E = experts, K = experts_per_token;
  if (T < 0 || E <= 0 || K <= 0 || K > E || logits.shape() != Shape{1, T, E}) {
    throw std::invalid_argument("OLMoE router geometry differs");
  }
#if MLX_VERSION_NUMERIC == 32002 || MLX_VERSION_NUMERIC == 32003
  if (T == 1 && E == 64 && K == 8 && logits.dtype() == float16 &&
      stream.device.type == Device::gpu) {
    // CustomKernelFunction captures source/options, not a stream. One
    // function per thread is bounded even if sessions create many streams;
    // every invocation explicitly carries its owning stream to MLX.
    static thread_local const fast::CustomKernelFunction kernel = [] {
      CompileOptions options;
      options.math_mode = MathMode::Safe;
      return fast::metal_kernel("trifuse_olmoe_router",
          {"router_logits"}, {"indices", "gates"}, kOLMoERouterSource,
          "", true, false, options);
    }();
    auto result = kernel({logits}, {{1, T, K}, {1, T, K}},
        {uint32, float16}, {32, 1, 1}, {32, 1, 1}, {},
        std::nullopt, false, stream);
    return {std::move(result.at(0)), std::move(result.at(1))};
  }
#endif
  const auto probabilities = softmax(logits, -1, true, stream);
  const auto order = argpartition(negative(probabilities, stream), K - 1, -1, stream);
  const auto indices = slice(order, {0, 0, 0}, {1, T, K}, stream);
  return {indices, take_along_axis(probabilities, indices, -1, stream)};
}

} // namespace executorch::backends::mlx
