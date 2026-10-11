// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
#pragma once

#include "MLXConstantAliasDiagnostics.h"

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace executorch::backends::mlx {

enum class MLXConstantPoolMode { Off, Observe, Reuse };
inline const char* mlx_constant_pool_mode_name(MLXConstantPoolMode mode) noexcept {
  switch (mode) {
    case MLXConstantPoolMode::Observe: return "observe";
    case MLXConstantPoolMode::Reuse: return "reuse";
    default: return "off";
  }
}
inline MLXConstantPoolMode parse_mlx_constant_pool_mode(const char* value) {
  if (!value || std::string_view(value) == "off") return MLXConstantPoolMode::Off;
  if (std::string_view(value) == "observe") return MLXConstantPoolMode::Observe;
  if (std::string_view(value) == "reuse") return MLXConstantPoolMode::Reuse;
  throw std::invalid_argument("TRIFUSE_MLX_CONSTANT_POOL must be off, observe, or reuse");
}

// Exact identity within one owner only. The source is kept live by a lease,
// never dereferenced by this CPU-only core and never included in receipts.
struct MLXConstantPoolKey {
  const void* source = nullptr;
  size_t source_bytes = 0, logical_bytes = 0;
  std::string name;
  std::vector<int> shape;
  int dtype = 0, device_type = 0, device_index = 0;
  bool operator==(const MLXConstantPoolKey& other) const {
    return source == other.source && source_bytes == other.source_bytes &&
        logical_bytes == other.logical_bytes && name == other.name &&
        shape == other.shape && dtype == other.dtype &&
        device_type == other.device_type && device_index == other.device_index;
  }
};
struct MLXConstantPoolKeyHash {
  size_t operator()(const MLXConstantPoolKey& key) const noexcept {
    size_t hash = std::hash<const void*>{}(key.source);
    const auto mix = [&hash](size_t value) { hash ^= value + size_t(0x9e3779b9) + (hash << 6) + (hash >> 2); };
    mix(key.source_bytes); mix(key.logical_bytes); mix(std::hash<std::string>{}(key.name));
    for (const int dimension : key.shape) mix(size_t(dimension));
    mix(size_t(key.dtype)); mix(size_t(key.device_type)); mix(size_t(key.device_index));
    return hash;
  }
};

enum class MLXConstantPoolFallback { None, Off, Device, ZeroCopy, InitChain, Storage };
struct MLXConstantPoolStats {
  uint64_t loads = 0, completed_loads = 0, incomplete_loads = 0, load_exception_exits = 0;
  uint64_t eligible_loads = 0, fallback_loads = 0;
  uint64_t eligible_tensors = 0, eligible_bytes = 0;
  uint64_t matched_tensors = 0, matched_bytes = 0;
  uint64_t cross_load_matched_tensors = 0, cross_load_matched_bytes = 0;
  uint64_t within_load_matched_tensors = 0, within_load_matched_bytes = 0;
  uint64_t cross_load_reused_tensors = 0, cross_load_reused_bytes = 0;
  uint64_t cross_load_same_identity_matches = 0, cross_load_different_identity_matches = 0, cross_load_unknown_identity_matches = 0;
  uint64_t reused_tensors = 0, reused_bytes = 0;
  uint64_t created_tensors = 0, created_bytes = 0;
  uint64_t same_identity_matches = 0, different_identity_matches = 0, unknown_identity_matches = 0;
  uint64_t fallback_tensors = 0, fallback_bytes = 0;
  uint64_t fallback_off_tensors = 0, fallback_device_tensors = 0;
  uint64_t fallback_zero_copy_tensors = 0, fallback_init_chain_tensors = 0, fallback_storage_tensors = 0;
  uint64_t exceptions = 0, diagnostic_errors = 0;
  uint64_t unique_keys = 0, unique_key_logical_bytes = 0;
  uint64_t unique_wrapper_count = 0, unique_wrapper_length_bytes = 0, unique_wrapper_allocated_bytes = 0;
  uint64_t allocation_available_wrappers = 0, allocation_unavailable_wrappers = 0, identity_unavailable = 0;
  bool allocation_probe_registered = false, allocation_dedup_complete = true, counter_overflow = false;
};

// A Value copy must retain its storage. Factory(lease) must capture the lease in
// that storage's lifetime, so returned graph values can outlive this pool.
// All operations occur on the exclusive owner/load thread, under the existing
// runtime graph lock. No process-global pointer cache or cross-owner sharing.
template <class Value>
class BasicMLXConstantPool {
 public:
  using Identity = const void* (*)(const Value&);
  using Stats = MLXConstantPoolStats;
  class Load {
    BasicMLXConstantPool* pool_;
    MLXConstantPoolFallback fallback_;
    int exceptions_at_start_;
    uint64_t ordinal_ = 0;
    bool completed_ = false;
   public:
    Load(BasicMLXConstantPool& pool, bool gpu, bool zero_copy, bool no_init_chain) noexcept
        : pool_(&pool), fallback_(pool.mode_ == MLXConstantPoolMode::Off ? MLXConstantPoolFallback::Off :
            !gpu ? MLXConstantPoolFallback::Device : !zero_copy ? MLXConstantPoolFallback::ZeroCopy :
            !no_init_chain ? MLXConstantPoolFallback::InitChain : MLXConstantPoolFallback::None),
          exceptions_at_start_(std::uncaught_exceptions()) {
      pool_->add(pool_->stats_.loads, 1);
      ordinal_ = pool_->stats_.loads;
      pool_->add(eligible() ? pool_->stats_.eligible_loads : pool_->stats_.fallback_loads, 1);
    }
    Load(const Load&) = delete;
    Load& operator=(const Load&) = delete;
    Load(Load&& other) noexcept
        : pool_(std::exchange(other.pool_, nullptr)), fallback_(other.fallback_),
          exceptions_at_start_(other.exceptions_at_start_), ordinal_(other.ordinal_), completed_(other.completed_) {}
    ~Load() noexcept {
      if (!pool_) return;
      pool_->add(completed_ ? pool_->stats_.completed_loads : pool_->stats_.incomplete_loads, 1);
      if (std::uncaught_exceptions() > exceptions_at_start_)
        pool_->add(pool_->stats_.load_exception_exits, 1);
    }
    bool eligible() const noexcept { return fallback_ == MLXConstantPoolFallback::None; }
    MLXConstantPoolFallback fallback_reason() const noexcept { return fallback_; }
    void mark_completed() noexcept { completed_ = true; }
    void fallback(uint64_t bytes, MLXConstantPoolFallback reason = MLXConstantPoolFallback::None) noexcept {
      pool_->record_fallback(bytes, reason == MLXConstantPoolFallback::None ? fallback_ : reason);
    }
    template <class Factory>
    Value acquire(const MLXConstantPoolKey& key, std::shared_ptr<void> lease, Factory&& factory) {
      if (!eligible()) throw std::logic_error("Constant pool acquire outside eligible load");
      return pool_->acquire(key, std::move(lease), std::forward<Factory>(factory), ordinal_);
    }
  };

  explicit BasicMLXConstantPool(MLXConstantPoolMode mode, Identity identity,
      MLXConstantAllocationProbe probe = mlx_constant_allocation_probe.load(std::memory_order_acquire))
      : mode_(mode), identity_(identity), probe_(probe) {
    if (mode != MLXConstantPoolMode::Off && mode != MLXConstantPoolMode::Observe && mode != MLXConstantPoolMode::Reuse)
      throw std::invalid_argument("Invalid constant pool mode");
    if (!identity) throw std::invalid_argument("Constant pool requires a wrapper identity function");
    stats_.allocation_probe_registered = probe != nullptr;
  }
  BasicMLXConstantPool(const BasicMLXConstantPool&) = delete;
  BasicMLXConstantPool& operator=(const BasicMLXConstantPool&) = delete;
  Load begin_load(bool gpu, bool zero_copy, bool no_init_chain) noexcept {
    return Load(*this, gpu, zero_copy, no_init_chain);
  }
  MLXConstantPoolMode mode() const noexcept { return mode_; }
  const Stats& stats() const noexcept { return stats_; }
  bool allocation_complete() const noexcept {
    return probe_ && stats_.allocation_dedup_complete && !stats_.identity_unavailable &&
        !stats_.counter_overflow && stats_.allocation_available_wrappers == stats_.unique_wrapper_count;
  }
  void write_receipt(FILE* file = stderr) const noexcept {
    if (!file) return;
    char line[4096];
    const int count = std::snprintf(line, sizeof(line),
      "TRIFUSE_MLX_CONSTANT_POOL mode=%s loads=%llu completed_loads=%llu incomplete_loads=%llu load_exception_exits=%llu "
      "eligible_loads=%llu fallback_loads=%llu eligible_tensors=%llu eligible_bytes=%llu "
      "matched_tensors=%llu matched_bytes=%llu reused_tensors=%llu reused_bytes=%llu created_tensors=%llu created_bytes=%llu "
      "cross_load_matched_tensors=%llu cross_load_matched_bytes=%llu within_load_matched_tensors=%llu within_load_matched_bytes=%llu "
      "cross_load_reused_tensors=%llu cross_load_reused_bytes=%llu "
      "same_identity_matches=%llu different_identity_matches=%llu unknown_identity_matches=%llu "
      "cross_load_same_identity_matches=%llu cross_load_different_identity_matches=%llu cross_load_unknown_identity_matches=%llu "
      "fallback_tensors=%llu fallback_bytes=%llu fallback_off_tensors=%llu fallback_device_tensors=%llu "
      "fallback_zero_copy_tensors=%llu fallback_init_chain_tensors=%llu fallback_storage_tensors=%llu "
      "exceptions=%llu diagnostic_errors=%llu unique_keys=%llu unique_key_logical_bytes=%llu "
      "unique_wrapper_count=%llu unique_wrapper_length_bytes=%llu unique_wrapper_allocated_bytes=%llu "
      "allocation_probe_registered=%d allocation_available_wrappers=%llu allocation_unavailable_wrappers=%llu "
      "identity_unavailable=%llu allocation_complete=%d counter_overflow=%d "
      "dedup_scope=this_owner_only physical_memory_measured=0\n",
      mlx_constant_pool_mode_name(mode_), ull(stats_.loads), ull(stats_.completed_loads), ull(stats_.incomplete_loads), ull(stats_.load_exception_exits),
      ull(stats_.eligible_loads), ull(stats_.fallback_loads), ull(stats_.eligible_tensors), ull(stats_.eligible_bytes),
      ull(stats_.matched_tensors), ull(stats_.matched_bytes), ull(stats_.reused_tensors), ull(stats_.reused_bytes), ull(stats_.created_tensors), ull(stats_.created_bytes),
      ull(stats_.cross_load_matched_tensors), ull(stats_.cross_load_matched_bytes), ull(stats_.within_load_matched_tensors), ull(stats_.within_load_matched_bytes),
      ull(stats_.cross_load_reused_tensors), ull(stats_.cross_load_reused_bytes),
      ull(stats_.same_identity_matches), ull(stats_.different_identity_matches), ull(stats_.unknown_identity_matches),
      ull(stats_.cross_load_same_identity_matches), ull(stats_.cross_load_different_identity_matches), ull(stats_.cross_load_unknown_identity_matches),
      ull(stats_.fallback_tensors), ull(stats_.fallback_bytes), ull(stats_.fallback_off_tensors), ull(stats_.fallback_device_tensors),
      ull(stats_.fallback_zero_copy_tensors), ull(stats_.fallback_init_chain_tensors), ull(stats_.fallback_storage_tensors),
      ull(stats_.exceptions), ull(stats_.diagnostic_errors), ull(stats_.unique_keys), ull(stats_.unique_key_logical_bytes),
      ull(stats_.unique_wrapper_count), ull(stats_.unique_wrapper_length_bytes), ull(stats_.unique_wrapper_allocated_bytes),
      int(stats_.allocation_probe_registered), ull(stats_.allocation_available_wrappers), ull(stats_.allocation_unavailable_wrappers),
      ull(stats_.identity_unavailable), int(allocation_complete()), int(stats_.counter_overflow));
    if (count > 0) std::fwrite(line, 1, size_t(count) < sizeof(line) ? size_t(count) : sizeof(line) - 1, file);
  }

 private:
  struct OwnedValue {
    // Value is destroyed first; source memory remains valid through release.
    std::shared_ptr<void> lease;
    Value value;
    const void* identity;
    uint64_t first_load;
    OwnedValue(std::shared_ptr<void> owner, Value v, const void* id, uint64_t load)
        : lease(std::move(owner)), value(std::move(v)), identity(id), first_load(load) {}
  };
  static unsigned long long ull(uint64_t value) noexcept { return static_cast<unsigned long long>(value); }
  void add(uint64_t& target, uint64_t value) noexcept {
    if (value > std::numeric_limits<uint64_t>::max() - target) {
      target = std::numeric_limits<uint64_t>::max(); stats_.counter_overflow = true;
    } else target += value;
  }
  void record_fallback(uint64_t bytes, MLXConstantPoolFallback reason) noexcept {
    add(stats_.fallback_tensors, 1); add(stats_.fallback_bytes, bytes);
    switch (reason) {
      case MLXConstantPoolFallback::Off: add(stats_.fallback_off_tensors, 1); break;
      case MLXConstantPoolFallback::Device: add(stats_.fallback_device_tensors, 1); break;
      case MLXConstantPoolFallback::ZeroCopy: add(stats_.fallback_zero_copy_tensors, 1); break;
      case MLXConstantPoolFallback::InitChain: add(stats_.fallback_init_chain_tensors, 1); break;
      default: add(stats_.fallback_storage_tensors, 1); break;
    }
  }
  void record_match(const void* original, const void* current, bool cross_load) noexcept {
    add(!original || !current ? stats_.unknown_identity_matches :
        original == current ? stats_.same_identity_matches : stats_.different_identity_matches, 1);
    if (cross_load) add(!original || !current ? stats_.cross_load_unknown_identity_matches :
        original == current ? stats_.cross_load_same_identity_matches : stats_.cross_load_different_identity_matches, 1);
  }
  void record_wrapper(const void* identity) noexcept {
    if (!identity) { add(stats_.identity_unavailable, 1); return; }
    if (!stats_.allocation_dedup_complete) return;
    try {
      if (!identities_.emplace(identity, true).second) return;
      add(stats_.unique_wrapper_count, 1);
      if (probe_) {
        const auto allocation = probe_(identity);
        if (allocation.available) {
          add(stats_.allocation_available_wrappers, 1);
          add(stats_.unique_wrapper_length_bytes, allocation.length_bytes);
          add(stats_.unique_wrapper_allocated_bytes, allocation.allocated_bytes);
        } else add(stats_.allocation_unavailable_wrappers, 1);
      }
    } catch (...) {
      stats_.allocation_dedup_complete = false; add(stats_.diagnostic_errors, 1);
    }
  }
  template <class Factory>
  Value acquire(const MLXConstantPoolKey& key, std::shared_ptr<void> lease, Factory&& factory, uint64_t ordinal) {
    add(stats_.eligible_tensors, 1); add(stats_.eligible_bytes, key.logical_bytes);
    try {
      if (!key.source || !lease || !key.logical_bytes || key.logical_bytes > key.source_bytes ||
          key.name.empty() || key.dtype < 0 || key.device_type < 0 || key.device_index < 0)
        throw std::invalid_argument("Invalid exact constant pool source key");
      for (const int dimension : key.shape)
        if (dimension <= 0) throw std::invalid_argument("Constant pool does not share empty/invalid tensor shapes");
      const auto found = entries_.find(key);
      const bool matched = found != entries_.end();
      const bool cross_load = matched && found->second->first_load != ordinal;
      if (matched) {
        add(stats_.matched_tensors, 1); add(stats_.matched_bytes, key.logical_bytes);
        add(cross_load ? stats_.cross_load_matched_tensors : stats_.within_load_matched_tensors, 1);
        add(cross_load ? stats_.cross_load_matched_bytes : stats_.within_load_matched_bytes, key.logical_bytes);
        if (mode_ == MLXConstantPoolMode::Reuse) {
          Value result = found->second->value;
          record_match(found->second->identity, identity_(result), cross_load);
          add(stats_.reused_tensors, 1); add(stats_.reused_bytes, key.logical_bytes);
          if (cross_load) { add(stats_.cross_load_reused_tensors, 1); add(stats_.cross_load_reused_bytes, key.logical_bytes); }
          return result;
        }
      }
      Value value = factory(lease);
      const void* identity = identity_(value);
      auto owned = std::make_shared<OwnedValue>(std::move(lease), std::move(value), identity, ordinal);
      // All created wrappers remain alive until owner teardown, so an opaque
      // identity cannot be freed/recycled under the observation/dedup table.
      retained_.push_back(owned);
      if (!matched) {
        try {
          entries_.emplace(key, owned);
        } catch (...) { retained_.pop_back(); throw; }
        add(stats_.unique_keys, 1); add(stats_.unique_key_logical_bytes, key.logical_bytes);
      } else record_match(found->second->identity, identity, cross_load);
      add(stats_.created_tensors, 1); add(stats_.created_bytes, key.logical_bytes);
      record_wrapper(identity);
      return owned->value;
    } catch (...) { add(stats_.exceptions, 1); throw; }
  }
  MLXConstantPoolMode mode_;
  Identity identity_;
  MLXConstantAllocationProbe probe_;
  Stats stats_;
  std::unordered_map<MLXConstantPoolKey, std::shared_ptr<OwnedValue>, MLXConstantPoolKeyHash> entries_;
  std::vector<std::shared_ptr<OwnedValue>> retained_;
  std::unordered_map<const void*, bool> identities_;
};

// Template keeps nested/exception/thread isolation testable without MLX.
// It owns the active pool strongly only for synchronous delegate loading.
template <class Pool>
inline std::shared_ptr<Pool>& mlx_constant_pool_load_context() noexcept {
  // Function-local TLS retains one context per type/thread across translation
  // units, without duplicate variable-template TLS initializer symbols on iOS.
  static thread_local std::shared_ptr<Pool> context;
  return context;
}
template <class Pool>
class ScopedMLXConstantPool {
  std::shared_ptr<Pool> previous_;
 public:
  explicit ScopedMLXConstantPool(std::shared_ptr<Pool> pool) noexcept
      : previous_(std::move(mlx_constant_pool_load_context<Pool>())) {
    mlx_constant_pool_load_context<Pool>() = std::move(pool);
  }
  ~ScopedMLXConstantPool() noexcept { mlx_constant_pool_load_context<Pool>() = std::move(previous_); }
  ScopedMLXConstantPool(const ScopedMLXConstantPool&) = delete;
  ScopedMLXConstantPool& operator=(const ScopedMLXConstantPool&) = delete;
};
template <class Pool>
std::shared_ptr<Pool> current_mlx_constant_pool_for_type() noexcept {
  return mlx_constant_pool_load_context<Pool>();
}

} // namespace executorch::backends::mlx
