#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <stdexcept>

namespace executorch::backends::mlx {

// Cached array references only. Packed weights remain packed on the GPU; the
// per-group bias is an affine dequantization offset, not an ordinary layer bias.
class NativeLinearWeight {
 public:
  static NativeLinearWeight dense(const ::mlx::core::array& weight,
                                  const ::mlx::core::Stream& stream) {
    using namespace ::mlx::core;
    need(weight.ndim() == 2 && (weight.dtype() == float16 || weight.dtype() == float32),
         "dense matrix must be FP16/FP32");
    return NativeLinearWeight(transpose(weight, {1, 0}, stream), std::nullopt,
                              std::nullopt, weight.shape(1), weight.dtype(), 0, 0);
  }
  static NativeLinearWeight affine(const ::mlx::core::array& packed,
                                   const ::mlx::core::array& scales,
                                   const ::mlx::core::array& biases,
                                   int output_width, int input_width,
                                   int bits, int group_size) {
    using namespace ::mlx::core;
    need((bits == 4 || bits == 8) &&
         (group_size == 32 || group_size == 64 || group_size == 128), "unsupported packing");
    need(input_width > 0 && output_width > 0 && input_width % group_size == 0 &&
         input_width % (32 / bits) == 0, "invalid logical geometry");
    need(packed.dtype() == uint32 && packed.shape() == Shape{output_width, input_width / (32 / bits)},
         "packed matrix differs from logical geometry");
    need(scales.shape() == Shape{output_width, input_width / group_size} &&
         biases.shape() == scales.shape() && biases.dtype() == scales.dtype() &&
         (scales.dtype() == float16 || scales.dtype() == float32), "invalid affine parameters");
    for (const auto* value : {&packed, &scales, &biases})
      need(value->is_available() && value->flags().row_contiguous,
           "packed parameters must already be available and row contiguous");
    return NativeLinearWeight(packed, scales, biases, input_width, scales.dtype(), bits, group_size);
  }
  ::mlx::core::array operator()(const ::mlx::core::array& x,
                                const ::mlx::core::Stream& stream) const {
    using namespace ::mlx::core;
    need(x.ndim() >= 1 && x.shape(-1) == input_width_ && x.dtype() == dtype_,
         "activation shape/dtype differs from linear parameters");
    if (!bits_) return matmul(x, weight_, stream);
    return quantized_matmul(x, weight_, *scales_, biases_, true,
                            group_size_, bits_, "affine", stream);
  }
 private:
  ::mlx::core::array weight_;
  std::optional<::mlx::core::array> scales_, biases_;
  int input_width_;
  ::mlx::core::Dtype dtype_;
  int bits_, group_size_;
  NativeLinearWeight(::mlx::core::array weight,
                     std::optional<::mlx::core::array> scales,
                     std::optional<::mlx::core::array> biases,
                     int width, ::mlx::core::Dtype dtype, int bits, int group)
      : weight_(std::move(weight)), scales_(std::move(scales)), biases_(std::move(biases)),
        input_width_(width), dtype_(dtype), bits_(bits), group_size_(group) {}
  static void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(std::string("native linear: ") + message);
  }
};
} // namespace executorch::backends::mlx
