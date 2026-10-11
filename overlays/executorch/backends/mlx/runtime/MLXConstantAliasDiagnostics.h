// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <initializer_list>
#include <string_view>
#include <unordered_map>

namespace executorch::backends::mlx {

struct MLXConstantBufferAllocation {
  bool available = false;
  uint64_t length_bytes = 0, allocated_bytes = 0;
};
using MLXConstantAllocationProbe = MLXConstantBufferAllocation (*)(const void* opaque_buffer) noexcept;
inline std::atomic<MLXConstantAllocationProbe> mlx_constant_allocation_probe{nullptr};

// The platform adapter may register this only for an exclusive request whose
// allocator's opaque buffer type it knows. Clear/restore after request drain.
// Generic runtime code never casts the identity or includes Metal headers.
inline MLXConstantAllocationProbe set_mlx_constant_allocation_probe(MLXConstantAllocationProbe probe) noexcept {
  return mlx_constant_allocation_probe.exchange(probe, std::memory_order_acq_rel);
}

inline bool mlx_constant_alias_diagnostics_enabled() noexcept {
  const char* value = std::getenv("TRIFUSE_MLX_CONSTANT_ALIAS_DIAGNOSTICS");
  return value && value[0] == '1' && value[1] == '\0';
}

// Match the actual OLMoE/Granite packed expert names, not router weights or
// arbitrary names containing "expert". Unrecognized architectures are other.
inline bool mlx_constant_is_known_expert(std::string_view name) noexcept {
  for (const auto marker : {std::string_view(".mlp.switch_mlp."),
                            std::string_view(".block_sparse_moe.switch_mlp.")}) {
    const auto at = name.rfind(marker);
    if (at == std::string_view::npos) continue;
    const auto prefix = name.substr(0, at);
    const auto layer = prefix.rfind("layers.");
    if (layer == std::string_view::npos || (layer && prefix[layer - 1] != '.')) continue;
    const auto digits = prefix.substr(layer + 7);
    if (digits.empty()) continue;
    bool numeric = true;
    for (const char c : digits) numeric &= c >= '0' && c <= '9';
    if (!numeric) continue;
    const auto tail = name.substr(at + marker.size());
    for (const auto projection : {std::string_view("gate_proj."),
                                  std::string_view("up_proj."), std::string_view("down_proj.")}) {
      if (tail.substr(0, projection.size()) != projection) continue;
      const auto field = tail.substr(projection.size());
      if (field == "weight" || field == "scales" || field == "biases") return true;
    }
  }
  return false;
}

// Diagnostic only: data-pointer equality is observed immediately after array
// construction. Neither equality nor an opaque Buffer identity proves Metal
// residency, allocation size, pinning, expert placement, or physical RAM use.
class MLXConstantAliasDiagnostics {
 public:
  struct Counts {
    uint64_t tensors = 0, logical_bytes = 0;
    uint64_t aliased_tensors = 0, aliased_bytes = 0;
    uint64_t copied_tensors = 0, copied_bytes = 0;
    uint64_t unknown_tensors = 0, unknown_bytes = 0;
  };

  explicit MLXConstantAliasDiagnostics(bool zero_copy_config) noexcept
      : invocation_(next_invocation_.fetch_add(1, std::memory_order_relaxed)),
        zero_copy_config_(zero_copy_config),
        allocation_probe_(mlx_constant_allocation_probe.load(std::memory_order_acquire)) {}
  MLXConstantAliasDiagnostics(const MLXConstantAliasDiagnostics&) = delete;
  MLXConstantAliasDiagnostics& operator=(const MLXConstantAliasDiagnostics&) = delete;

  // CPU-only entry for contract tests. Addresses are retained only as opaque
  // keys until this load ends and are never printed or dereferenced here.
  void observe(std::string_view name, uint64_t bytes, const void* original,
               const void* constructed, const void* buffer_identity, bool available) noexcept {
    auto& group = mlx_constant_is_known_expert(name) ? experts_ : other_;
    add_counts(total_, bytes, original, constructed, available);
    add_counts(group, bytes, original, constructed, available);
    if (!available || !buffer_identity) {
      add(identity_unavailable_, 1);
      return;
    }
    if (dedup_failed_) return;
    try {
      auto [it, inserted] = buffers_.try_emplace(buffer_identity, bytes);
      if (inserted) {
        add(unique_buffer_max_tensor_bytes_, bytes);
        if (allocation_probe_) {
          const auto allocation = allocation_probe_(buffer_identity);
          if (allocation.available) {
            add(allocation_available_buffers_, 1);
            add(allocation_length_bytes_, allocation.length_bytes);
            add(allocation_allocated_bytes_, allocation.allocated_bytes);
          } else {
            add(allocation_unavailable_buffers_, 1);
          }
        }
      } else if (bytes > it->second) {
        add(unique_buffer_max_tensor_bytes_, bytes - it->second);
        it->second = bytes;
      }
    } catch (...) {
      // Instrumentation must not turn allocation pressure into inference failure.
      dedup_failed_ = true;
      add(diagnostic_errors_, 1);
    }
  }

  template <class Array>
  void observe_array(std::string_view name, const void* original, const Array& array) noexcept {
    uint64_t bytes = 0;
    try {
      bytes = array.nbytes();
      if (!array.is_available()) {
        observe(name, bytes, original, nullptr, nullptr, false);
        return; // Never eval, wait, or request data from an unavailable array.
      }
      const void* constructed = array.template data<uint8_t>();
      observe(name, bytes, original, constructed, array.buffer().ptr(), true);
    } catch (...) {
      add(diagnostic_errors_, 1);
      observe(name, bytes, original, nullptr, nullptr, false);
    }
  }

  void mark_load_completed() noexcept { completed_ = true; }
  const Counts& total() const noexcept { return total_; }
  const Counts& experts() const noexcept { return experts_; }
  const Counts& other() const noexcept { return other_; }
  size_t unique_buffer_count() const noexcept { return buffers_.size(); }
  uint64_t unique_buffer_max_tensor_bytes() const noexcept { return unique_buffer_max_tensor_bytes_; }
  bool dedup_complete() const noexcept { return !dedup_failed_ && !identity_unavailable_ && !counter_overflow_; }
  uint64_t allocation_available_buffers() const noexcept { return allocation_available_buffers_; }
  uint64_t allocation_length_bytes() const noexcept { return allocation_length_bytes_; }
  uint64_t allocation_allocated_bytes() const noexcept { return allocation_allocated_bytes_; }
  bool allocation_complete() const noexcept {
    return allocation_probe_ && dedup_complete() && allocation_available_buffers_ == buffers_.size();
  }

  ~MLXConstantAliasDiagnostics() noexcept {
    // One line per load invocation, including an interrupted load. Logical
    // byte totals can repeat across prefill/decode program loads. The deduped
    // payload is the maximum observed tensor size per opaque buffer identity,
    // not that buffer's allocated size or the union of its view addresses.
    char line[4096];
    const int size = std::snprintf(line, sizeof(line),
        "TRIFUSE_MLX_CONSTANT_ALIAS invocation=%llu cfg_zero_copy=%d load_completed=%d "
        "tensors=%llu logical_bytes=%llu aliased_tensors=%llu aliased_bytes=%llu "
        "copied_tensors=%llu copied_bytes=%llu unknown_tensors=%llu unknown_bytes=%llu "
        "expert_tensors=%llu expert_logical_bytes=%llu expert_aliased_tensors=%llu expert_aliased_bytes=%llu "
        "expert_copied_tensors=%llu expert_copied_bytes=%llu expert_unknown_tensors=%llu expert_unknown_bytes=%llu "
        "other_tensors=%llu other_logical_bytes=%llu other_aliased_tensors=%llu other_aliased_bytes=%llu "
        "other_copied_tensors=%llu other_copied_bytes=%llu other_unknown_tensors=%llu other_unknown_bytes=%llu "
        "unique_buffer_count=%zu unique_buffer_max_tensor_bytes=%llu dedup_complete=%d "
        "allocation_probe_registered=%d allocation_available_buffers=%llu allocation_unavailable_buffers=%llu "
        "allocation_length_bytes=%llu allocation_allocated_bytes=%llu allocation_complete=%d "
        "identity_unavailable=%llu diagnostic_errors=%llu counter_overflow=%d "
        "dedup_scope=this_load_call_only sums_across_loads_may_repeat=1 physical_memory_measured=0 "
        "expert_name_rule=numeric_layer_switch_mlp_gate_up_down_QSB\n",
        ull(invocation_), int(zero_copy_config_), int(completed_),
        ull(total_.tensors), ull(total_.logical_bytes), ull(total_.aliased_tensors), ull(total_.aliased_bytes),
        ull(total_.copied_tensors), ull(total_.copied_bytes), ull(total_.unknown_tensors), ull(total_.unknown_bytes),
        ull(experts_.tensors), ull(experts_.logical_bytes), ull(experts_.aliased_tensors), ull(experts_.aliased_bytes),
        ull(experts_.copied_tensors), ull(experts_.copied_bytes), ull(experts_.unknown_tensors), ull(experts_.unknown_bytes),
        ull(other_.tensors), ull(other_.logical_bytes), ull(other_.aliased_tensors), ull(other_.aliased_bytes),
        ull(other_.copied_tensors), ull(other_.copied_bytes), ull(other_.unknown_tensors), ull(other_.unknown_bytes),
        buffers_.size(), ull(unique_buffer_max_tensor_bytes_), int(dedup_complete()),
        int(allocation_probe_ != nullptr), ull(allocation_available_buffers_), ull(allocation_unavailable_buffers_),
        ull(allocation_length_bytes_), ull(allocation_allocated_bytes_), int(allocation_complete()),
        ull(identity_unavailable_), ull(diagnostic_errors_), int(counter_overflow_));
    if (size > 0) {
      const size_t length = size_t(size) < sizeof(line) ? size_t(size) : sizeof(line) - 1;
      std::fwrite(line, 1, length, stderr);
    }
  }

 private:
  static unsigned long long ull(uint64_t value) noexcept { return static_cast<unsigned long long>(value); }
  void add(uint64_t& target, uint64_t value) noexcept {
    if (value > std::numeric_limits<uint64_t>::max() - target) {
      target = std::numeric_limits<uint64_t>::max();
      counter_overflow_ = true;
    } else target += value;
  }
  void add_counts(Counts& counts, uint64_t bytes, const void* source,
                  const void* destination, bool available) noexcept {
    add(counts.tensors, 1);
    add(counts.logical_bytes, bytes);
    if (!available || !bytes || !source || !destination) {
      add(counts.unknown_tensors, 1); add(counts.unknown_bytes, bytes);
    } else if (source == destination) {
      add(counts.aliased_tensors, 1); add(counts.aliased_bytes, bytes);
    } else {
      add(counts.copied_tensors, 1); add(counts.copied_bytes, bytes);
    }
  }
  inline static std::atomic<uint64_t> next_invocation_{1};
  const uint64_t invocation_;
  const bool zero_copy_config_;
  const MLXConstantAllocationProbe allocation_probe_;
  bool completed_ = false, dedup_failed_ = false, counter_overflow_ = false;
  Counts total_, experts_, other_;
  uint64_t unique_buffer_max_tensor_bytes_ = 0, identity_unavailable_ = 0, diagnostic_errors_ = 0;
  uint64_t allocation_available_buffers_ = 0, allocation_unavailable_buffers_ = 0;
  uint64_t allocation_length_bytes_ = 0, allocation_allocated_bytes_ = 0;
  std::unordered_map<const void*, uint64_t> buffers_;
};

} // namespace executorch::backends::mlx
