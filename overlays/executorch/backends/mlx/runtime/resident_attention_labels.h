#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace executorch::backends::mlx {
class MLXCache;

// Explicit offline diagnostic. Uniform native-Llama GQA geometry is supplied
// by the admitted native execution, never inferred from a model name.
struct ResidentAttentionLabelConfig {
  int block_tokens = 64;
  std::vector<int> layers;

  void validate_for(int layer_count, int token_count) const {
    if (block_tokens != 64 || token_count != 1 || layers.empty())
      throw std::invalid_argument("resident attention labels: native T1 and block64 required");
    for (size_t i = 0; i < layers.size(); ++i) {
      if (layers[i] < 0 || layers[i] >= layer_count ||
          std::find(layers.begin(), layers.begin() + i, layers[i]) != layers.begin() + i)
        throw std::invalid_argument("resident attention labels: invalid or repeated selected layer");
    }
  }
};

struct AttentionLabelBatch {
  int query_position = 0;
  // Both lengths include this call's consumed token. An offline collector must
  // separately bind the earlier student coverage and future application window.
  int covered_length = 0;
  int full_length = 0;
  int block_tokens = 64;
  int block_count = 0;
  int query_heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  std::vector<int> layers;
  // Row-major FP32 [layers.size(), kv_heads, block_count]. Softmax each query
  // head, max within its contiguous GQA group, then sum absolute block64.
  // The last partial block sums only real history; its missing cells are zero.
  std::vector<float> values;
  static constexpr const char* source = "native_post_rope_qk_recomputed_fp32_attention";
  static constexpr const char* source_precision =
      "actual native activation/KV QK cast to FP32 before score matmul and softmax";
  static constexpr bool offline_only = true;
  static constexpr bool fused_kernel_probabilities = false;
  static constexpr bool causal_deployment_advice = false;
};

class AttentionLabelTicket {
 public:
  virtual ~AttentionLabelTicket() = default;
  // One consumer. Waits outside the MLX global mutex; reads and releases MLX
  // arrays under that mutex. Any error invalidates the enclosing observation.
  virtual AttentionLabelBatch wait_and_read() = 0;
};

// Thread-local explicit opt-in, bound to one serialized fresh dense owner.
// Rejected for T>1, partial/cache-only prefill, compacted or sparse history, and
// caches that have previously executed sparse attention. No default graph work.
// Callback is invoked after submission outside the MLX global mutex and must
// only retain/enqueue the ticket, never wait/read or reenter a model owner.
// The caller retains model/cache/ET inputs through normal forward completion,
// then drains all tickets before release. Unconsumed unfinished/error payloads
// are quarantined instead of blocking or releasing uncertain GPU dependencies.
class ResidentAttentionLabelScope {
 public:
  using Callback = std::function<void(std::shared_ptr<AttentionLabelTicket>)>;
  ResidentAttentionLabelScope(MLXCache* expected_cache, ResidentAttentionLabelConfig config, Callback on_ticket);
  ~ResidentAttentionLabelScope();
  ResidentAttentionLabelScope(const ResidentAttentionLabelScope&) = delete;
  ResidentAttentionLabelScope& operator=(const ResidentAttentionLabelScope&) = delete;
  static ResidentAttentionLabelScope* current();
  MLXCache* cache() const { return cache_; }
  const ResidentAttentionLabelConfig& config() const { return config_; }
  void publish(std::shared_ptr<AttentionLabelTicket> ticket) const;

 private:
  MLXCache* cache_;
  ResidentAttentionLabelConfig config_;
  Callback on_ticket_;
};
} // namespace executorch::backends::mlx
