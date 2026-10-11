#pragma once

#include <cstdint>
#include <memory>
#include <mlx/array.h>

namespace executorch::backends::mlx {
class MLXCache;
class MLXBackend;
class NativeGraniteMoEPrefill;

// An exclusive owner holds this context through drain/quarantine. The private
// stream is shared by the admitted Granite policy's prefill and T1 delegates.
// Destruction releases metadata only; it never evaluates or drains lazy work.
class ResidentMoEPolicyContext {
 public:
  explicit ResidentMoEPolicyContext(MLXCache* cache);
  ~ResidentMoEPolicyContext();
  void synchronize();
 private:
  friend class MLXBackend;
  friend class ResidentMoEPolicyScope;
  struct Storage;
  std::unique_ptr<Storage> storage_;
};

class ResidentMoEPolicyScope {
 public:
  explicit ResidentMoEPolicyScope(std::shared_ptr<ResidentMoEPolicyContext> context);
  ~ResidentMoEPolicyScope();
  ResidentMoEPolicyScope(const ResidentMoEPolicyScope&) = delete;
  ResidentMoEPolicyScope& operator=(const ResidentMoEPolicyScope&) = delete;
 private:
  friend class MLXBackend;
  static ResidentMoEPolicyScope* current();
  std::shared_ptr<ResidentMoEPolicyContext> context_;
};

// Only MLXBackend can create this admitted input. Ordinary host-token forward
// retains its availability/range validation; this value stays on the GPU.
class ResidentGreedyDeviceInput {
  friend class MLXBackend;
  friend class NativeGraniteMoEPrefill;
  explicit ResidentGreedyDeviceInput(::mlx::core::array value) : value_(std::move(value)) {}
  ::mlx::core::array value_;
};

// A submitted prediction, not a completed append. Destruction releases arrays
// under the MLX graph mutex; the owner must drain before releasing its leases.
class ResidentGreedyTicket {
 public:
  ~ResidentGreedyTicket();
  int64_t wait_token() const;
 private:
  friend class MLXBackend;
  friend class ResidentGreedySubmissionScope;
  struct Storage;
  ResidentGreedyTicket();
  std::unique_ptr<Storage> storage_;
};

// Private Granite T1 submit-only mode. The previous prediction is fed directly
// to embedding on GPU, while position and owner admission stay host-controlled.
// No ET output bytes are populated. At most one invocation per scope is allowed.
class ResidentGreedySubmissionScope {
 public:
  ResidentGreedySubmissionScope(MLXCache* cache, int vocabulary, int position,
      std::shared_ptr<ResidentGreedyTicket> input, std::shared_ptr<void> input_lease,
      bool initial_prediction = false);
  ~ResidentGreedySubmissionScope();
  ResidentGreedySubmissionScope(const ResidentGreedySubmissionScope&) = delete;
  ResidentGreedySubmissionScope& operator=(const ResidentGreedySubmissionScope&) = delete;
  std::shared_ptr<ResidentGreedyTicket> submitted_ticket() const;
  // Retain before Module.forward, including a partially submitted failure.
  std::shared_ptr<ResidentGreedyTicket> cleanup_ticket() const { return output_; }
 private:
  friend class MLXBackend;
  static ResidentGreedySubmissionScope* current();
  MLXCache* cache_;
  int vocabulary_, position_;
  bool invoked_ = false;
  bool initial_prediction_ = false;
  std::shared_ptr<ResidentGreedyTicket> input_, output_;
  std::shared_ptr<void> input_lease_;
};

// Same serialized owner/thread requirement as idle allocator maintenance.
// This only clears unused allocator buffers, never evaluates lazy model roots.
void clear_greedy_allocator_cache();

struct ResidentGreedyResult {
  int64_t token = -1;
  int vocabulary = 0;
  // True only for the audited native Granite full-history path, after both
  // scalar completion events and all successful-path maintenance have passed.
  // Every canonical K/V write is an ancestor of those completed logits.
  bool kv_dependencies_completed = false;
};

// Private synchronous scalar-output mode for one native GPU T1 invocation.
// Use together with NativeLlamaPrefillScope on the same serialized owner/thread.
// The backend checks the complete FP32 logits row for finite values on the GPU,
// selects its first maximum, and waits for the token and finite-check scalars.
// The audited Granite full-history path completes every K/V write through the
// logits dependency graph; other architectures also submit and wait on all K/V
// roots. Only an explicit result receipt permits the bridge to omit its extra
// storage evaluation. Submission alone never publishes a completed result.
// Module.forward's ET output bytes are NOT populated and must never be read.
// Ordinary forward/logits behavior is unchanged outside this scope. This is
// scalar-output optimization, not asynchronous token generation or lookahead.
class ResidentGreedyOutputScope {
 public:
  ResidentGreedyOutputScope(MLXCache* expected_cache, int vocabulary);
  ~ResidentGreedyOutputScope();
  ResidentGreedyOutputScope(const ResidentGreedyOutputScope&) = delete;
  ResidentGreedyOutputScope& operator=(const ResidentGreedyOutputScope&) = delete;

  bool completed() const { return completed_; }
  ResidentGreedyResult result() const;

 private:
  friend class MLXBackend;
  static ResidentGreedyOutputScope* current();
  void begin_invocation();
  void finish(ResidentGreedyResult result);
  void fail() noexcept;

  MLXCache* cache_;
  int vocabulary_;
  bool invoked_ = false;
  bool completed_ = false;
  ResidentGreedyResult result_;
};
} // namespace executorch::backends::mlx
