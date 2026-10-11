#pragma once

#include <cstddef>

namespace executorch::backends::mlx {
class MLXCache;

enum class NativeModelArchitecture { StandardLlamaTied, GPTNeoXParallel, GraniteMoE, OLMoE };

inline bool is_native_moe_policy_architecture(NativeModelArchitecture architecture) {
  return architecture == NativeModelArchitecture::GraniteMoE ||
      architecture == NativeModelArchitecture::OLMoE;
}
enum class NativeQuantizationLayout { Uniform, AffinePerProjection };

// Explicit architecture configuration bound to validated model metadata.
// Common cache geometry is shared; each adapter admits its own math/weights.
struct NativeModelConfig {
  int layers = 0;
  int query_heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int vocabulary = 0;
  int capacity = 0;
  float rms_norm_eps = 0;
  float rope_theta = 0;
  int linear_quantization_bits = 0;
  int linear_quantization_group_size = 0;
  NativeModelArchitecture architecture = NativeModelArchitecture::StandardLlamaTied;
  int rotary_dimensions = 0;
  float layer_norm_eps = 0;
  NativeQuantizationLayout quantization_layout = NativeQuantizationLayout::Uniform;
  int num_experts = 0;
  int experts_per_token = 0;
  float embedding_multiplier = 1;
  float residual_multiplier = 1;
  float attention_multiplier = 1;
  float logits_scaling = 1;
  bool tie_word_embeddings = true;
};

// Keep the resident ABI's historical source name while using one config for
// both prefill and decode. Default construction retains existing Llama math.
using NativeLlamaPrefillConfig = NativeModelConfig;

// Pure admission before an owner creates pending invocations. Loaded-program
// structure and weights are separately checked by the chosen native adapter.
void validate_native_model_configuration(const NativeModelConfig& config);

// Blocks new backend submissions while waiting for already-submitted work on
// every MLX stream. Does not evaluate lazy graphs or reclaim/change cache memory.
// Call outside a backend execution/global-mutex scope; failures propagate.
void drain_submitted_work();

// Covers a serialized prefill batch or one opted-in T1 decode call. The name
// remains for compatibility. Cache-only submission is a separate scope below.
// Unknown/unsupported programs fail rather than silently using another path.
// active_tokens=0 retains the complete static input. A positive count opts in
// to exactly one native GPU/full-history call: only the leading tokens
// participate, while the fixed ET ABI remains intact with zero tail padding.
// A partial cache-only call additionally requires its explicit submission opt-in.
class NativeLlamaPrefillScope {
 public:
  NativeLlamaPrefillScope(MLXCache* cache, NativeLlamaPrefillConfig config,
                         int active_tokens = 0);
  ~NativeLlamaPrefillScope();
  NativeLlamaPrefillScope(const NativeLlamaPrefillScope&) = delete;
  NativeLlamaPrefillScope& operator=(const NativeLlamaPrefillScope&) = delete;
  static NativeLlamaPrefillScope* current();
  MLXCache* cache() const { return cache_; }
  const NativeLlamaPrefillConfig& config() const { return config_; }
  int active_tokens() const { return active_tokens_; }
  void mark_active_tokens_dispatched();
 private:
  MLXCache* cache_;
  NativeLlamaPrefillConfig config_;
  int active_tokens_ = 0;
  bool active_tokens_dispatched_ = false;
};

// Private opt-in for a resident owner that retains input storage until a later
// synchronous drain. During this scope Module.forward's output bytes are NOT
// produced and must never be read. Ordinary forward semantics are unchanged.
// A scope permits exactly one delegate invocation, on the expected cache.
class ResidentPrefillSubmission {
 public:
  explicit ResidentPrefillSubmission(MLXCache* cache, bool allow_partial_native = false);
  ~ResidentPrefillSubmission();
  ResidentPrefillSubmission(const ResidentPrefillSubmission&) = delete;
  ResidentPrefillSubmission& operator=(const ResidentPrefillSubmission&) = delete;
  bool submitted() const { return submitted_; }
  static ResidentPrefillSubmission* current();
  MLXCache* cache() const { return cache_; }
  bool allow_partial_native() const { return allow_partial_native_; }
  void mark_submitted();
 private:
  MLXCache* cache_;
  bool allow_partial_native_ = false;
  bool submitted_ = false;
};
} // namespace executorch::backends::mlx
