#pragma once

// Pinned MLX 0.32.3 affine KV attention for one decode query, B=1, Hq=Hkv=16,
// D=128. Every supplied token is visible; the caller owns RoPE and causal
// placement. Uses native MLX ops without materializing a dequantized KV cache.
#include <mlx/mlx.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace trifuse_cold_kv_experiment {
namespace mx = ::mlx::core;
inline constexpr int kHeads = 16, kDim = 128, kGroupSize = 64, kBits = 8;

class ColdKVAttention {
 public:
  // Codes [1,16,C,32] uint32; scales/biases [1,16,C,2] FP16.
  // Hot K/V [1,16,H,128], query/output [1,16,1,128], all FP16.
  // Empty cold or hot is accepted, but not both. Hot inputs may be lazy strided
  // views: pass the actual arrays so dependencies on new KV writes are retained.
  // Construction is lazy; this function neither evaluates nor synchronizes.
  mx::array apply(const mx::array& q, const mx::array& cold_k,
      const mx::array& cold_ks, const mx::array& cold_kb,
      const mx::array& cold_v, const mx::array& cold_vs,
      const mx::array& cold_vb, const mx::array& hot_k,
      const mx::array& hot_v, mx::Stream stream,
      float attention_scale = 1.0f / std::sqrt(128.0f)) const {
    if (stream.device.type != mx::Device::gpu ||
        !std::isfinite(attention_scale) || attention_scale <= 0.0f)
      throw std::invalid_argument("affine KV attention requires GPU and finite positive scale");
    if (cold_k.ndim() != 4 || hot_k.ndim() != 4)
      throw std::invalid_argument("affine cold/hot KV must have four dimensions");
    const int c = cold_k.shape(2), h = hot_k.shape(2);
    if (c < 0 || h < 0 || c > 65536 || h > 65536 || c + h == 0)
      throw std::invalid_argument("invalid affine cold/hot token counts");
    require(q, {1, 16, 1, 128}, mx::float16);
    require(cold_k, {1, 16, c, 32}, mx::uint32);
    require(cold_v, {1, 16, c, 32}, mx::uint32);
    for (const auto* x : {&cold_ks, &cold_kb, &cold_vs, &cold_vb})
      require(*x, {1, 16, c, 2}, mx::float16);
    require(hot_k, {1, 16, h, 128}, mx::float16);
    require(hot_v, {1, 16, h, 128}, mx::float16);

    // Match MLX-LM: scale the FP16 query before the two score products.
    const auto query = mx::multiply(q, mx::array(attention_scale, q.dtype()), stream);
    std::vector<mx::array> scores;
    if (c) scores.push_back(mx::quantized_matmul(query, cold_k, cold_ks, cold_kb,
        true, kGroupSize, kBits, "affine", stream));
    if (h) scores.push_back(mx::matmul(query, mx::swapaxes(hot_k, -1, -2, stream), stream));
    const auto logits = scores.size() == 1 ? scores.front() : mx::concatenate(scores, -1, stream);
    // A single denominator across both segments is essential. Independently
    // normalized cold/hot attentions cannot simply be added or averaged.
    const auto probabilities = mx::softmax(logits, -1, true, stream);
    std::vector<mx::array> contributions;
    if (c) {
      const auto p = h ? mx::slice(probabilities, {0,0,0,0}, {1,16,1,c}, stream)
                       : probabilities;
      contributions.push_back(mx::quantized_matmul(p, cold_v, cold_vs, cold_vb,
          false, kGroupSize, kBits, "affine", stream));
    }
    if (h) {
      const auto p = c ? mx::slice(probabilities, {0,0,0,c}, {1,16,1,c+h}, stream)
                       : probabilities;
      contributions.push_back(mx::matmul(p, hot_v, stream));
    }
    const auto output = contributions.size() == 1 ? contributions.front()
        : mx::add(contributions[0], contributions[1], stream);
    return mx::astype(output, mx::float16, stream);
  }

 private:
  static void require(const mx::array& a, const mx::Shape& shape, mx::Dtype dtype) {
    if (a.shape() != shape || a.dtype() != dtype)
      throw std::invalid_argument("affine KV attention shape/dtype mismatch");
  }
};
} // namespace trifuse_cold_kv_experiment
