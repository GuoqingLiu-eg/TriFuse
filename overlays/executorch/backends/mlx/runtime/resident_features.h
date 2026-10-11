#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace executorch::backends::mlx {
class MLXCache;

struct ResidentFeatureConfig {
  int block_tokens = 64;
  std::vector<int> layers;

  void validate_for(int layer_count, int token_count) const {
    if (block_tokens <= 0 || token_count <= 1 || token_count % block_tokens ||
        layers.empty()) {
      throw std::invalid_argument("resident features: nonempty native prefill blocks/layers required");
    }
    for (size_t i = 0; i < layers.size(); ++i) {
      if (layers[i] < 0 || layers[i] >= layer_count ||
          std::find(layers.begin(), layers.begin() + i, layers[i]) != layers.begin() + i) {
        throw std::invalid_argument("resident features: invalid or repeated selected layer");
      }
    }
  }
};

struct FeatureBatch {
  int absolute_start = 0;
  int token_count = 0;
  int block_tokens = 0;
  std::vector<int> layers;
  // FP32, row-major [layers.size(), token_count / block_tokens, 6].
  // h is the input RMSNorm result; k is new, pre-RoPE K; v is new V.
  // All reductions accumulate in FP32 over this call's new tokens only.
  static constexpr size_t feature_count = 6;
  static constexpr std::array<const char*, feature_count> feature_names{
      "h_mean_square", "h_max_abs", "k_mean_square", "k_max_abs",
      "v_mean_square", "v_max_abs"};
  std::vector<float> values;
};

class FeatureTicket {
 public:
  virtual ~FeatureTicket() = default;
  // Exactly one consumer. Waits for this submitted summary outside the MLX
  // global mutex, then reads owned finite FP32 values. Errors must invalidate
  // the enclosing observation/request; they are not a successful empty batch.
  virtual FeatureBatch wait_and_read() = 0;
};

// Explicit, thread-local opt-in for a serialized native T>1 prefill scope.
// The callback runs after async submission, outside the MLX global mutex. It
// must only enqueue the ticket, never wait/read it or reenter a model owner.
// The caller retains model/cache/ET inputs until ordinary prefill completion
// and drains queued tickets before releasing them. No KV is exported here.
// Abandoned tickets never wait in their destructor: resources with unknown
// completion are quarantined until process exit, so bound admission upstream.
class ResidentFeatureScope {
 public:
  using Callback = std::function<void(std::shared_ptr<FeatureTicket>)>;
  ResidentFeatureScope(MLXCache* expected_cache, ResidentFeatureConfig config, Callback on_ticket);
  ~ResidentFeatureScope();
  ResidentFeatureScope(const ResidentFeatureScope&) = delete;
  ResidentFeatureScope& operator=(const ResidentFeatureScope&) = delete;
  static ResidentFeatureScope* current();
  MLXCache* cache() const { return cache_; }
  const ResidentFeatureConfig& config() const { return config_; }
  void publish(std::shared_ptr<FeatureTicket> ticket) const;

 private:
  MLXCache* cache_;
  ResidentFeatureConfig config_;
  Callback on_ticket_;
};
} // namespace executorch::backends::mlx
