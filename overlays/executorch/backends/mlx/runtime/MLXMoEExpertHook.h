// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>

#include <atomic>
#include <functional>
#include <optional>
#include <span>
#include <utility>

namespace executorch::backends::mlx {

// Opt-in prefill experiment boundary. Hidden states and original routes are
// exposed here; the model owner, attention and KV cache remain in the runtime.
class NativeMoEExpertHook {
 public:
  using Array = ::mlx::core::array;
  using Stream = ::mlx::core::Stream;

  // Inputs are parallel one-dimensional token/expert indices in caller route
  // order, unless experts_sorted is explicitly true (caller guarantees
  // nondecreasing expert IDs). The result is {expert-sorted output [N,H], inverse [N]}, where
  // output[inverse[i]] belongs to input route i. N == 0 is supported.
  // This callable is borrowed for execute() only. Invoke it on the execution
  // thread, and do not retain it or use it in a background worker.
  using GPUCallable = std::function<std::pair<Array, Array>(
      const Array& token_indices, const Array& expert_ids, bool experts_sorted)>;

  virtual ~NativeMoEExpertHook() = default;

  // Optional, frozen plan metadata for hint-only weight-page experiments.
  // This borrowed span is used on the exclusive execution thread and copied
  // into job metadata before any worker submission. Empty means no exclusions.
  // It is NOT the actual routed expert set: selected experts may overflow to
  // GPU, and all their original weights must remain available on demand.
  virtual std::span<const int> planned_ane_experts_for_prefetch(int /*layer*/) const {
    return {};
  }

  // Hidden/residual are [1,T,H]; indices/gates are [1,T,K]. A result must
  // include the weighted expert reduction and residual, in hidden's dtype
  // and residual's shape. Empty means use the existing GPU implementation.
  // No evaluation barrier is inserted by the runtime: a hook must account
  // for its own synchronization and keep asynchronous borrowed data alive.
  virtual std::optional<Array> execute(
      int layer, int T, const Array& hidden, const Array& residual,
      const Array& indices, const Array& gates, const Stream& stream,
      const GPUCallable& gpu) = 0;
};

// One shared slot across translation units, including the statically linked
// backend and its application. The caller owns the hook and may register it
// only around a single exclusive benchmark request. Clear after that request
// and its workers finish, before destroying the hook. Atomic publication is
// not concurrent-request support and does not extend the object's lifetime.
inline std::atomic<NativeMoEExpertHook*> native_moe_expert_hook{nullptr};

inline void set_native_moe_expert_hook(NativeMoEExpertHook* hook) {
  native_moe_expert_hook.store(hook, std::memory_order_release);
}

inline NativeMoEExpertHook* get_native_moe_expert_hook() {
  return native_moe_expert_hook.load(std::memory_order_acquire);
}

} // namespace executorch::backends::mlx
