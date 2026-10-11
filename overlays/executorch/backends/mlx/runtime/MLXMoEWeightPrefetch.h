// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
#pragma once

#include <cerrno>
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace executorch::backends::mlx {

enum class NativeMoEWeightPrefetchMode { Off, WillNeed, Coroutine };
enum class NativeMoEWeightPrefetchScope { Full, GPUComplement };

inline NativeMoEWeightPrefetchMode native_moe_weight_prefetch_mode() {
  const char* value = std::getenv("TRIFUSE_MOE_WEIGHT_PREFETCH");
  if (!value || !*value || std::string_view(value) == "off") return NativeMoEWeightPrefetchMode::Off;
  if (std::string_view(value) == "willneed") return NativeMoEWeightPrefetchMode::WillNeed;
  if (std::string_view(value) == "coroutine") return NativeMoEWeightPrefetchMode::Coroutine;
  throw std::invalid_argument("TRIFUSE_MOE_WEIGHT_PREFETCH must be off, willneed or coroutine");
}

inline bool native_moe_weight_prefetch_enabled() {
  return native_moe_weight_prefetch_mode() != NativeMoEWeightPrefetchMode::Off;
}

inline NativeMoEWeightPrefetchScope native_moe_weight_prefetch_scope(NativeMoEWeightPrefetchMode mode) {
  const char* value = std::getenv("TRIFUSE_MOE_WEIGHT_PREFETCH_SCOPE");
  if (!value || std::string_view(value) == "full") return NativeMoEWeightPrefetchScope::Full;
  if (std::string_view(value) == "gpu-complement" && mode != NativeMoEWeightPrefetchMode::Off) {
    return NativeMoEWeightPrefetchScope::GPUComplement;
  }
  throw std::invalid_argument(
      "TRIFUSE_MOE_WEIGHT_PREFETCH_SCOPE must be full or gpu-complement (requires active prefetch)");
}

inline const char* native_moe_weight_prefetch_scope_name(NativeMoEWeightPrefetchScope scope) {
  return scope == NativeMoEWeightPrefetchScope::GPUComplement ? "gpu-complement" : "full";
}

inline int native_moe_weight_prefetch_layer_budget(NativeMoEWeightPrefetchMode mode) {
  const char* value = std::getenv("TRIFUSE_MOE_WEIGHT_PREFETCH_LAYER_BUDGET");
  if (!value) return 0;
  if (mode != NativeMoEWeightPrefetchMode::Coroutine) {
    throw std::invalid_argument("TRIFUSE_MOE_WEIGHT_PREFETCH_LAYER_BUDGET requires coroutine mode");
  }
  const std::string_view text(value);
  int budget = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), budget);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || budget <= 0) {
    throw std::invalid_argument("TRIFUSE_MOE_WEIGHT_PREFETCH_LAYER_BUDGET must be a positive int");
  }
  return budget;
}

struct MoEWeightPrefetchLimits {
  // Zero is the internal unlimited value; explicitly supplied environment
  // values must be positive. Bytes count page-aligned advice, not physical IO.
  size_t max_bytes = 0;
  uint64_t max_host_ms = 0;
};

inline MoEWeightPrefetchLimits native_moe_weight_prefetch_limits(NativeMoEWeightPrefetchMode mode) {
  const auto read = [mode](const char* name) -> uint64_t {
    const char* value = std::getenv(name);
    if (!value) return 0;
    if (mode != NativeMoEWeightPrefetchMode::Coroutine) {
      throw std::invalid_argument(std::string(name) + " requires coroutine mode");
    }
    const std::string_view text(value);
    uint64_t limit = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), limit);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !limit) {
      throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return limit;
  };
  const uint64_t mib = read("TRIFUSE_MOE_WEIGHT_PREFETCH_MAX_MIB");
  constexpr size_t bytes_per_mib = 1024 * 1024;
  if (mib > std::numeric_limits<size_t>::max() / bytes_per_mib) {
    throw std::invalid_argument("TRIFUSE_MOE_WEIGHT_PREFETCH_MAX_MIB overflows byte count");
  }
  return {size_t(mib) * bytes_per_mib, read("TRIFUSE_MOE_WEIGHT_PREFETCH_MAX_HOST_MS")};
}

inline std::vector<int> moe_weight_prefetch_layer_selection(int eligible_layers, int layer_budget) {
  if (eligible_layers < 0 || layer_budget < 0) {
    throw std::invalid_argument("MoE prefetch eligible layers and internal budget must be nonnegative");
  }
  const int count = layer_budget ? std::min(layer_budget, eligible_layers) : eligible_layers;
  std::vector<int> selected;
  selected.reserve(size_t(count));
  // Select the center rank of each equal-width interval; zero budget means all.
  for (int i = 0; i < count; ++i) {
    selected.push_back(1 + int((2 * int64_t(i) + 1) * eligible_layers / (2 * int64_t(count))));
  }
  return selected;
}

struct MoEWeightPageRange { uintptr_t begin = 0; size_t bytes = 0; };

inline MoEWeightPageRange moe_weight_page_range(
    uintptr_t data, size_t bytes, size_t page_size) {
  const auto maximum = std::numeric_limits<uintptr_t>::max();
  if (!data || !bytes || !page_size || bytes > maximum - data) return {};
  const uintptr_t begin = data - data % page_size;
  const uintptr_t end = data + bytes;
  const size_t padding = (page_size - end % page_size) % page_size;
  if (padding > maximum - end) return {};
  return {begin, size_t(end + padding - begin)};
}

// Only the synchronous host hint is timed here. A successful hint is not a
// residency guarantee; any subsequent paging cost stays in the request.
struct MoEWeightPrefetchStats {
  size_t layers = 0, calls = 0, succeeded = 0, failed = 0, invalid_ranges = 0;
  size_t bytes = 0;
  int first_errno = 0;
  double host_ms = 0, max_hint_ms = 0;

  void advise(const void* data, size_t length) {
    const auto start = std::chrono::steady_clock::now();
    const long page_size = ::sysconf(_SC_PAGESIZE);
    const auto range = moe_weight_page_range(
        reinterpret_cast<uintptr_t>(data), length,
        page_size > 0 ? size_t(page_size) : 0);
    if (!range.bytes) {
      ++invalid_ranges;
    } else {
      ++calls;
      bytes += range.bytes;
      if (::madvise(reinterpret_cast<void*>(range.begin), range.bytes, MADV_WILLNEED) == 0) {
        ++succeeded;
      } else {
        const int error = errno;
        ++failed;
        if (!first_errno) first_errno = error;
      }
    }
    const double hint_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    host_ms += hint_ms;
    max_hint_ms = std::max(max_hint_ms, hint_ms);
  }

};

struct MoEWeightByteRange { const void* data = nullptr; size_t bytes = 0; };
struct MoEExpertHintRun { size_t begin = 0, end = 0; }; // Half-open expert indices.
struct ExpertHintLayout {
  size_t expert_count = 0;
  std::vector<MoEExpertHintRun> included_runs;
};

inline std::shared_ptr<const ExpertHintLayout> make_moe_expert_hint_layout(
    int experts, std::span<const int> excluded) {
  if (experts <= 0) throw std::invalid_argument("MoE hint expert count must be positive");
  // An empty selection keeps the existing nine ranges and allocates nothing.
  if (excluded.empty()) return {};
  std::vector<int> ids(excluded.begin(), excluded.end());
  std::sort(ids.begin(), ids.end());
  if (ids.front() < 0 || ids.back() >= experts ||
      std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
    throw std::invalid_argument("MoE hint excluded experts must be distinct valid IDs");
  }
  auto layout = std::make_shared<ExpertHintLayout>();
  layout->expert_count = size_t(experts);
  layout->included_runs.reserve(ids.size() + 1);
  size_t begin = 0;
  for (const int excluded_id : ids) {
    const size_t end = size_t(excluded_id);
    if (begin < end) layout->included_runs.push_back({begin, end});
    begin = end + 1;
  }
  if (begin < size_t(experts)) layout->included_runs.push_back({begin, size_t(experts)});
  return layout;
}

struct MoEWeightPrefetchJob {
  std::array<MoEWeightByteRange, 9> tensors{};
  // Null preserves the old full-range path without heap allocation. Only the
  // immutable interval metadata is shared; constant pointers remain borrowed.
  std::shared_ptr<const ExpertHintLayout> layout;
  auto& operator[](size_t i) { return tensors[i]; }
  const auto& operator[](size_t i) const { return tensors[i]; }
  auto begin() { return tensors.begin(); }
  auto end() { return tensors.end(); }
  auto begin() const { return tensors.begin(); }
  auto end() const { return tensors.end(); }
  constexpr size_t size() const { return 9; }
  auto data() { return tensors.data(); }
  auto data() const { return tensors.data(); }
  void fill(const MoEWeightByteRange& range) { tensors.fill(range); }
};

inline void validate_moe_weight_prefetch_job(const MoEWeightPrefetchJob& job) {
  if (!job.layout) return;
  const auto& layout = *job.layout;
  if (!layout.expert_count) throw std::invalid_argument("MoE hint layout has no experts");
  size_t end = 0;
  bool first = true;
  for (const auto& run : layout.included_runs) {
    if (run.begin >= run.end || run.end > layout.expert_count || (!first && run.begin <= end)) {
      throw std::invalid_argument("MoE hint runs must be sorted, disjoint and merged");
    }
    end = run.end;
    first = false;
  }
  for (const auto& range : job) {
    if (range.bytes % layout.expert_count) {
      throw std::invalid_argument("MoE hint tensor must have a contiguous integral expert axis");
    }
  }
}

struct MoEWeightHintLogicalBytes { uint64_t all = 0, hint = 0; };
inline MoEWeightHintLogicalBytes moe_weight_hint_logical_bytes(const MoEWeightPrefetchJob& job) {
  validate_moe_weight_prefetch_job(job);
  size_t included = 0;
  if (job.layout) for (const auto& run : job.layout->included_runs) included += run.end - run.begin;
  MoEWeightHintLogicalBytes result;
  for (const auto& range : job) {
    const size_t hint = job.layout ? range.bytes / job.layout->expert_count * included : range.bytes;
    if (range.bytes > std::numeric_limits<uint64_t>::max() - result.all ||
        hint > std::numeric_limits<uint64_t>::max() - result.hint) {
      throw std::invalid_argument("MoE hint logical byte sum overflows");
    }
    result.all += range.bytes;
    result.hint += hint;
  }
  return result;
}

// CPU-only page-range iterator. It lazily generates runs and removes page
// overlap within each tensor only. Page rounding can cover a boundary portion
// of an excluded expert; logical byte savings are not physical IO savings.
class MoEWeightHintCursor {
 public:
  MoEWeightHintCursor(const MoEWeightPrefetchJob& job, size_t page_size)
      : job_(job), page_size_(page_size) {
    validate_moe_weight_prefetch_job(job);
  }
  bool next(MoEWeightPageRange& output) {
    while (tensor_ < job_.size()) {
      const auto& tensor = job_[tensor_];
      if (!job_.layout) {
        ++tensor_;
        output = moe_weight_page_range(reinterpret_cast<uintptr_t>(tensor.data), tensor.bytes, page_size_);
      } else {
        const auto& layout = *job_.layout;
        if (run_ == layout.included_runs.size()) {
          ++tensor_; run_ = 0; previous_end_ = 0; continue;
        }
        const auto run = layout.included_runs[run_++];
        const size_t stride = tensor.bytes / layout.expert_count;
        const size_t offset = stride * run.begin, length = stride * (run.end - run.begin);
        const uintptr_t base = reinterpret_cast<uintptr_t>(tensor.data);
        output = base && offset <= std::numeric_limits<uintptr_t>::max() - base
            ? moe_weight_page_range(base + offset, length, page_size_) : MoEWeightPageRange{};
        if (output.bytes) {
          const uintptr_t end = output.begin + output.bytes;
          const uintptr_t begin = std::max(output.begin, previous_end_);
          previous_end_ = end;
          if (begin >= end) continue;
          output = {begin, size_t(end - begin)};
        }
      }
      if (output.bytes) return true;
      ++invalid_ranges_;
    }
    return false;
  }
  size_t invalid_ranges() const { return invalid_ranges_; }
 private:
  const MoEWeightPrefetchJob& job_;
  size_t page_size_, tensor_ = 0, run_ = 0, invalid_ranges_ = 0;
  uintptr_t previous_end_ = 0;
};

// One job at a time, with no queue. The run-local owner joins before its
// borrowed constant pointers can expire; this thread never calls MLX.
class MoEWeightPrefetchWorker {
 public:
  explicit MoEWeightPrefetchWorker(NativeMoEWeightPrefetchScope scope = NativeMoEWeightPrefetchScope::Full)
      : scope_(scope), worker_([this] { work(); }) {}
  MoEWeightPrefetchWorker(const MoEWeightPrefetchWorker&) = delete;
  MoEWeightPrefetchWorker& operator=(const MoEWeightPrefetchWorker&) = delete;

  bool try_submit(const MoEWeightPrefetchJob& job) {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || busy_) {
      ++skipped_layers_;
      return false;
    }
    const auto logical = moe_weight_hint_logical_bytes(job);
    job_ = job;
    submitted_logical_all_bytes_ += logical.all;
    submitted_logical_hint_bytes_ += logical.hint;
    if (job.layout) ++planned_expert_filtered_jobs_;
    busy_ = true;
    pending_ = true;
    ++submitted_layers_;
    lock.unlock();
    ready_.notify_one();
    return true;
  }

  ~MoEWeightPrefetchWorker() {
    const auto start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    ready_.notify_one();
    worker_.join();
    const double join_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr,
        "TRIFUSE_MOE_WEIGHT_PREFETCH mode=willneed ahead=1 host=worker scope=%s "
        "submitted_logical_all_bytes=%llu submitted_logical_hint_bytes=%llu planned_expert_filtered_jobs=%zu "
        "submitted_layers=%zu skipped_layers=%zu layers=%zu calls=%zu "
        "succeeded=%zu failed=%zu invalid_ranges=%zu hint_bytes=%zu "
        "first_errno=%d worker_host_ms=%.3f join_ms=%.3f\n",
        native_moe_weight_prefetch_scope_name(scope_),
        static_cast<unsigned long long>(submitted_logical_all_bytes_),
        static_cast<unsigned long long>(submitted_logical_hint_bytes_), planned_expert_filtered_jobs_,
        submitted_layers_, skipped_layers_, stats_.layers, stats_.calls,
        stats_.succeeded, stats_.failed, stats_.invalid_ranges, stats_.bytes,
        stats_.first_errno, stats_.host_ms, join_ms);
  }

 private:
  void work() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      ready_.wait(lock, [this] { return pending_ || stop_; });
      if (!pending_) return;
      const auto job = job_;
      pending_ = false;
      lock.unlock();
      ++stats_.layers;
      if (!job.layout) {
        for (const auto& range : job) stats_.advise(range.data, range.bytes);
      } else {
        const long page_size = ::sysconf(_SC_PAGESIZE);
        MoEWeightHintCursor cursor(job, page_size > 0 ? size_t(page_size) : 0);
        MoEWeightPageRange range;
        while (cursor.next(range)) stats_.advise(reinterpret_cast<const void*>(range.begin), range.bytes);
        stats_.invalid_ranges += cursor.invalid_ranges();
      }
      lock.lock();
      busy_ = false;
    }
  }

  const NativeMoEWeightPrefetchScope scope_;
  uint64_t submitted_logical_all_bytes_ = 0, submitted_logical_hint_bytes_ = 0;
  size_t planned_expert_filtered_jobs_ = 0;
  std::mutex mutex_;
  std::condition_variable ready_;
  MoEWeightPrefetchJob job_;
  bool busy_ = false, pending_ = false, stop_ = false;
  size_t submitted_layers_ = 0, skipped_layers_ = 0;
  MoEWeightPrefetchStats stats_;
  std::thread worker_;
};

// Coroutine frames belong exclusively to this worker. The host only publishes
// its layer position and copies pointer/length pairs into an empty slot.
class MoEWeightPrefetchCoroutineWorker {
 public:
  explicit MoEWeightPrefetchCoroutineWorker(int eligible_layers = 0, int layer_budget = 0,
                                          MoEWeightPrefetchLimits limits = {},
                                          NativeMoEWeightPrefetchScope scope = NativeMoEWeightPrefetchScope::Full)
      : eligible_layers_(eligible_layers), requested_layer_budget_(layer_budget),
        selected_layer_ids_(moe_weight_prefetch_layer_selection(eligible_layers, layer_budget)),
        selected_layer_ids_text_(format_layer_ids(selected_layer_ids_)),
        limits_(limits), scope_(scope),
        worker_([this] { work(); }) {}
  MoEWeightPrefetchCoroutineWorker(const MoEWeightPrefetchCoroutineWorker&) = delete;
  MoEWeightPrefetchCoroutineWorker& operator=(const MoEWeightPrefetchCoroutineWorker&) = delete;

  void advance_layer(int layer) {
    current_layer_.store(layer, std::memory_order_release);
    ready_.notify_one();
  }

  // The host calls this once per eligible layer, before collecting its ranges.
  bool consider_layer(int layer) {
    if (layer <= 0 || layer > eligible_layers_) {
      throw std::invalid_argument("MoE prefetch layer is outside the eligible range");
    }
    ++considered_layers_;
    const bool selected = std::binary_search(selected_layer_ids_.begin(), selected_layer_ids_.end(), layer);
    if (!selected) ++skipped_by_budget_;
    return selected;
  }

  bool try_submit(int layer, const MoEWeightPrefetchJob& job) {
    if (budget_exhausted_.load(std::memory_order_acquire)) {
      ++skipped_layers_;
      ++limit_skipped_layers_;
      return false;
    }
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    // The worker can exhaust its budget between the first check and locking.
    if (budget_exhausted_.load(std::memory_order_acquire)) {
      ++skipped_layers_;
      ++limit_skipped_layers_;
      return false;
    }
    if (lock.owns_lock() && layer > current_layer_.load(std::memory_order_acquire)) {
      for (auto& slot : slots_) {
        if (slot.layer == layer) {
          ++skipped_layers_;
          return false;
        }
      }
      for (auto& slot : slots_) {
        if (slot.layer < 0) {
          const auto logical = moe_weight_hint_logical_bytes(job);
          slot.layer = layer;
          slot.job = job;
          submitted_logical_all_bytes_ += logical.all;
          submitted_logical_hint_bytes_ += logical.hint;
          if (job.layout) ++planned_expert_filtered_jobs_;
          ++submitted_layers_;
          lock.unlock();
          ready_.notify_one();
          return true;
        }
      }
    }
    ++skipped_layers_;
    return false;
  }

  ~MoEWeightPrefetchCoroutineWorker() {
    const auto start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    ready_.notify_one();
    worker_.join();
    const double join_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr,
        "TRIFUSE_MOE_WEIGHT_PREFETCH mode=coroutine ahead=1 host=worker scope=%s "
        "submitted_logical_all_bytes=%llu submitted_logical_hint_bytes=%llu planned_expert_filtered_jobs=%zu "
        "capacity_layers=2 chunk_bytes=%zu eligible_layers=%d requested_layer_budget=%d "
        "layer_budget=%zu selected_layers=%zu considered_layers=%zu skipped_by_budget=%zu "
        "selected_layer_ids=%s submitted_layers=%zu skipped_layers=%zu "
        "completed_layers=%zu cancelled_layers=%zu expired_layers=%zu "
        "shutdown_cancelled_layers=%zu failed_layers=%zu "
        "max_hint_bytes=%zu max_host_ms=%llu max_hint_ms=%.3f host_overshoot_ms=%.3f "
        "budget_exhausted=%d limit_stop_reason=%s limit_skipped_layers=%zu "
        "budget_cancelled_layers=%zu layers=%zu calls=%zu "
        "succeeded=%zu failed=%zu invalid_ranges=%zu hint_bytes=%zu "
        "first_errno=%d worker_host_ms=%.3f join_ms=%.3f\n",
        native_moe_weight_prefetch_scope_name(scope_),
        static_cast<unsigned long long>(submitted_logical_all_bytes_),
        static_cast<unsigned long long>(submitted_logical_hint_bytes_), planned_expert_filtered_jobs_,
        chunk_bytes_, eligible_layers_, requested_layer_budget_, selected_layer_ids_.size(),
        selected_layer_ids_.size(), considered_layers_, skipped_by_budget_, selected_layer_ids_text_.c_str(),
        submitted_layers_, skipped_layers_, completed_layers_,
        cancelled_layers_, expired_layers_, shutdown_cancelled_layers_, failed_layers_,
        limits_.max_bytes, static_cast<unsigned long long>(limits_.max_host_ms), stats_.max_hint_ms,
        limits_.max_host_ms ? std::max(0.0, stats_.host_ms - double(limits_.max_host_ms)) : 0.0,
        int(budget_exhausted_.load(std::memory_order_acquire)), limit_stop_reason(),
        limit_skipped_layers_, budget_cancelled_layers_,
        stats_.layers, stats_.calls, stats_.succeeded, stats_.failed,
        stats_.invalid_ranges, stats_.bytes, stats_.first_errno, stats_.host_ms, join_ms);
  }

 private:
  // Only the worker modifies limit flags/stats. The owner observes the single
  // atomic exhaustion flag, and reads the final details only after join.
  bool check_limits() {
    if (limits_.max_bytes && (stats_.bytes >= limits_.max_bytes ||
        (page_size_ && limits_.max_bytes - stats_.bytes < page_size_))) {
      byte_limit_reached_ = true;
    }
    if (limits_.max_host_ms && stats_.host_ms >= double(limits_.max_host_ms)) {
      host_limit_reached_ = true;
    }
    if (byte_limit_reached_ || host_limit_reached_) {
      budget_exhausted_.store(true, std::memory_order_release);
      return true;
    }
    return false;
  }

  const char* limit_stop_reason() const {
    if (byte_limit_reached_ && host_limit_reached_) return "bytes_and_host_ms";
    if (byte_limit_reached_) return "bytes";
    if (host_limit_reached_) return "host_ms";
    return "none";
  }

  static std::string format_layer_ids(const std::vector<int>& layers) {
    if (layers.empty()) return "none";
    std::string result;
    for (const int layer : layers) {
      if (!result.empty()) result += ',';
      result += std::to_string(layer);
    }
    return result;
  }

  struct Task {
    struct promise_type {
      bool failed = false;
      Task get_return_object() { return Task{std::coroutine_handle<promise_type>::from_promise(*this)}; }
      std::suspend_always initial_suspend() noexcept { return {}; }
      std::suspend_always final_suspend() noexcept { return {}; }
      void return_void() noexcept {}
      void unhandled_exception() noexcept { failed = true; }
    };
    std::coroutine_handle<promise_type> handle{};
    Task() = default;
    explicit Task(std::coroutine_handle<promise_type> value) : handle(value) {}
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    Task(Task&& other) noexcept : handle(std::exchange(other.handle, {})) {}
    Task& operator=(Task&& other) noexcept {
      if (handle) handle.destroy();
      handle = std::exchange(other.handle, {});
      return *this;
    }
    ~Task() { if (handle) handle.destroy(); }
  };
  struct Slot {
    int layer = -1;
    MoEWeightPrefetchJob job;
    Task task;
  };

  Task hint_chunks(MoEWeightPrefetchJob job) {
    std::array<MoEWeightPageRange, 9> ranges;
    size_t ranges_left = 0;
    for (size_t i = 0; i < job.size(); ++i) {
      ranges[i] = moe_weight_page_range(
          reinterpret_cast<uintptr_t>(job[i].data), job[i].bytes, page_size_);
      if (ranges[i].bytes) ++ranges_left;
      else ++stats_.invalid_ranges;
    }
    for (const auto& range : ranges) {
      if (!range.bytes) continue;
      --ranges_left;
      size_t offset = 0;
      while (offset < range.bytes) {
        if (check_limits()) {
          // Leave this unfinished frame suspended for budget cancellation;
          // returning normally here would incorrectly count it as complete.
          co_await std::suspend_always{};
          co_return;
        }
        size_t bytes = std::min(chunk_bytes_, range.bytes - offset);
        if (limits_.max_bytes) {
          const size_t remaining = limits_.max_bytes - stats_.bytes;
          bytes = std::min(bytes, remaining - remaining % page_size_);
        }
        stats_.advise(reinterpret_cast<const void*>(range.begin + offset), bytes);
        offset += bytes;
        // A running syscall cannot be interrupted. Publish any resulting time
        // overshoot immediately, so no later chunk/job is admitted as work.
        check_limits();
        if (offset < range.bytes || ranges_left) co_await std::suspend_always{};
      }
    }
  }

  Task hint_filtered_chunks(MoEWeightPrefetchJob job) {
    MoEWeightHintCursor cursor(job, page_size_);
    MoEWeightPageRange range;
    size_t recorded_invalid = 0;
    const auto next = [&](MoEWeightPageRange& value) {
      const bool found = cursor.next(value);
      stats_.invalid_ranges += cursor.invalid_ranges() - recorded_invalid;
      recorded_invalid = cursor.invalid_ranges();
      return found;
    };
    bool more = next(range);
    while (more) {
      MoEWeightPageRange following;
      more = next(following);
      size_t offset = 0;
      while (offset < range.bytes) {
        if (check_limits()) {
          co_await std::suspend_always{};
          co_return;
        }
        size_t bytes = std::min(chunk_bytes_, range.bytes - offset);
        if (limits_.max_bytes) {
          const size_t remaining = limits_.max_bytes - stats_.bytes;
          bytes = std::min(bytes, remaining - remaining % page_size_);
        }
        stats_.advise(reinterpret_cast<const void*>(range.begin + offset), bytes);
        offset += bytes;
        check_limits();
        if (offset < range.bytes || more) co_await std::suspend_always{};
      }
      range = following;
    }
  }

  void work() {
    const long page_size = ::sysconf(_SC_PAGESIZE);
    page_size_ = page_size > 0 ? size_t(page_size) : 0;
    constexpr size_t target_chunk_bytes = 4 * 1024 * 1024;
    chunk_bytes_ = page_size_ ? std::max(page_size_, target_chunk_bytes / page_size_ * page_size_) : 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      ready_.wait(lock, [this] { return stop_ || budget_exhausted_.load(std::memory_order_acquire) ||
                                      slots_[0].layer >= 0 || slots_[1].layer >= 0; });
      const int current_layer = current_layer_.load(std::memory_order_acquire);
      const bool limited = budget_exhausted_.load(std::memory_order_acquire);
      Slot* selected = nullptr;
      for (auto& slot : slots_) {
        if (slot.layer < 0) continue;
        if (limited || stop_ || slot.layer < current_layer) {
          ++cancelled_layers_;
          if (limited) ++budget_cancelled_layers_;
          else if (stop_) ++shutdown_cancelled_layers_;
          else ++expired_layers_;
          slot.task = Task{};
          slot.layer = -1;
        } else if (!selected || slot.layer < selected->layer) {
          selected = &slot;
        }
      }
      if (stop_ || limited) return;
      if (!selected) continue;
      if (!selected->task.handle) {
        ++stats_.layers;
        try {
          selected->task = selected->job.layout ? hint_filtered_chunks(selected->job) : hint_chunks(selected->job);
        } catch (...) {
          ++failed_layers_;
          ++cancelled_layers_;
          selected->layer = -1;
          continue;
        }
      }
      lock.unlock();
      selected->task.handle.resume();
      lock.lock();
      if (selected->task.handle.done()) {
        if (selected->task.handle.promise().failed) {
          ++failed_layers_;
          ++cancelled_layers_;
        } else {
          ++completed_layers_;
        }
        selected->task = Task{};
        selected->layer = -1;
      }
    }
  }

  const int eligible_layers_ = 0, requested_layer_budget_ = 0;
  const std::vector<int> selected_layer_ids_;
  const std::string selected_layer_ids_text_;
  const MoEWeightPrefetchLimits limits_;
  const NativeMoEWeightPrefetchScope scope_;
  uint64_t submitted_logical_all_bytes_ = 0, submitted_logical_hint_bytes_ = 0;
  size_t planned_expert_filtered_jobs_ = 0;
  size_t considered_layers_ = 0, skipped_by_budget_ = 0;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::atomic<int> current_layer_{-1};
  std::atomic<bool> budget_exhausted_{false};
  std::array<Slot, 2> slots_;
  bool stop_ = false;
  size_t submitted_layers_ = 0, skipped_layers_ = 0;
  size_t completed_layers_ = 0, cancelled_layers_ = 0, expired_layers_ = 0;
  size_t shutdown_cancelled_layers_ = 0, failed_layers_ = 0;
  size_t limit_skipped_layers_ = 0, budget_cancelled_layers_ = 0;
  bool byte_limit_reached_ = false, host_limit_reached_ = false;
  size_t page_size_ = 0, chunk_bytes_ = 0;
  MoEWeightPrefetchStats stats_;
  std::thread worker_;
};

} // namespace executorch::backends::mlx
