#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <array>
#include <limits>

// Independent cold-state codec experiment. Build .mm with ARC and CoreML /
// Foundation; macOS 13+ / iOS 16+. CPUAndNeuralEngine allows CPU fallback.
// No operation here accesses, replaces, or releases the live GPU KV cache.
class ColdKVEncoder {
 public:
  static constexpr size_t kHeads = 16;
  static constexpr size_t kTokens = 256;
  static constexpr size_t kDimensions = 128;
  static constexpr size_t kVectors = kHeads * kTokens;
  static constexpr size_t kElements = kVectors * kDimensions;
  static constexpr size_t kGroupSize = 64;
  static constexpr size_t kBits = 8;
  static constexpr size_t kGroupsPerVector = kDimensions / kGroupSize;
  static constexpr size_t kWordsPerVector = kDimensions / 4;
  static constexpr const char* kFormat = "mlx_affine_uint32_g64_b8";

  struct Identity {
    uint64_t request_epoch = 0;
    uint64_t block_id = 0;
    int layer = 0;
    int start_position = 0;
  };
  struct QuantizedTarget {
    // Pointers already include this block's token offset. Capacities are the
    // remaining allocation extents FROM these pointers, in the named units.
    uint32_t* data = nullptr;
    void* scales = nullptr;
    void* biases = nullptr;
    size_t data_capacity_words = 0;
    size_t scales_capacity_groups = 0;
    size_t biases_capacity_groups = 0;
    size_t vector_stride_words = kWordsPerVector;
    size_t vector_stride_groups = kGroupsPerVector;
    size_t head_stride_words = kTokens * kWordsPerVector;
    size_t head_stride_groups = kTokens * kGroupsPerVector;
  };
  struct OutputTarget {
    Identity identity;
    QuantizedTarget keys;
    QuantizedTarget values;
    // Optional CPU-storage lease. When null, the external owner MUST keep all
    // allocations alive and inaccessible to readers through successful take,
    // failure take, or finish()/destruction join. Never release MLX objects on
    // this worker: the runtime keeps their owner itself. No descriptor or
    // mutable alias may change after submission. Failed writes are unpublished.
    std::shared_ptr<void> owner;
  };
  struct Snapshot {
    Identity identity;
    // Owning raw IEEE FP16 bit patterns, [head][token][dimension], each
    // kElements entries. The caller must freeze ALL mutable aliases before
    // submission. shared_ptr<const> alone cannot prevent external mutation.
    std::vector<uint16_t> keys;
    std::vector<uint16_t> values;
    std::shared_ptr<const OutputTarget> output;
  };
  struct Quantized {
    // Both outputs own their bytes; they outlive the encoder independently.
    // MLX affine g64/b8: data[(head*256+token)*32+channel/4].
    // Each uint32 packs four unsigned codes, earlier channel in low 8 bits.
    // FP16-bit scales/biases[(head*256+token)*2+channel/64]. Decode: q*scale+bias.
    std::vector<uint32_t> data;
    std::vector<uint16_t> scales;
    std::vector<uint16_t> biases;
  };
  struct TensorTiming {
    double input_copy_ms = 0; // FP16 layout copy, scoped input access; no transform
    double predict_ms = 0;    // synchronous CoreML call; includes internal copies
    double output_read_pack_ms = 0; // scoped bounds read + affine CPU packing
    double total_ms = 0;      // allocation and validation included
    uint64_t logical_input_bytes = 0;
    uint64_t logical_output_read_bytes = 0;
    uint64_t explicit_output_copy_bytes = 0; // pack directly from scoped output
    uint64_t stored_bytes = 0;
  };
  struct Result {
    Identity identity;
    Quantized keys;
    Quantized values;
    TensorTiming key_timing;
    TensorTiming value_timing;
    uint64_t submitted_monotonic_ns = 0;
    uint64_t started_monotonic_ns = 0;
    uint64_t ended_monotonic_ns = 0;
    double queue_wait_ms = 0;
    double encode_ms = 0; // two predictions plus packing; excludes queue wait
    float transport_scale = 1;
    bool post_unscale_fp16_rounding = false;
    bool actual_ane_execution_verified = false;
    std::string format = kFormat;
    uint64_t gpu_kv_bytes_released = 0;
    // Direct output owns no vectors and no target lease. The caller owns the
    // destination. A successful take synchronizes every worker write; no
    // partial output may be published after a worker exception.
    bool direct_output = false;
  };
  struct Preparation {
    bool cpu_only = false;
    double compile_ms = 0;
    double load_and_input_allocation_ms = 0;
    double total_ms = 0;
    bool compiled_package = false;
    bool cpu_fallback_allowed = true;
    bool actual_ane_execution_verified = false;
    std::string format = kFormat;
  };
  enum class State { Idle, Busy, Ready, Failed, Closed };
  struct Poll {
    State state = State::Idle;
    std::shared_ptr<const Result> result;
    std::exception_ptr error;
  };

  // Compile/load/input allocation happen synchronously, before request timing.
  // Expects affine_bounds from export_affine_codec.py: frozen_state FP16
  // [1,128,1,4096] -> bounds FP16 [1,4,1,4096] (min0,min1,max0,max1).
  explicit ColdKVEncoder(const std::string& packageOrCompiledPath);
  ~ColdKVEncoder(); // stops admission and joins an accepted encode
  ColdKVEncoder(const ColdKVEncoder&) = delete;
  ColdKVEncoder& operator=(const ColdKVEncoder&) = delete;

  const Preparation& preparation() const;
  // No queue buildup: at most one queued/running/untaken result. Invalid shape
  // throws; busy/closed returns false without waiting. Snapshot stays owned
  // until prediction/packing completes, including exception paths.
  bool trySubmit(std::shared_ptr<const Snapshot> snapshot);
  // Nonblocking TAKE. Ready/Failed is consumed exactly once. Mutex contention
  // reports Busy. A worker failure closes admission; no implicit retry.
  Poll poll();
  // Owner-thread convenience for a serialized transaction. Requires idle,
  // submits to the same worker, waits on its condition variable, then takes
  // the result or rethrows its exception. No extra model/predict path.
  std::shared_ptr<const Result> encode(std::shared_ptr<const Snapshot> snapshot);
  // Stop accepting and join queued/in-flight work. Does not interrupt CoreML.
  // A final untaken result/error remains available to poll(). Idempotent.
  // Serialize finish/destruction with other owner APIs; never call on worker.
  void finish();

  // Pure-CPU preflight. Conservative addressed-span checks intentionally
  // reject interleaved/overlapping outputs; the intended layout is six
  // independent arrays. This cannot prove a caller's allocation capacities.
  static void validateOutputTarget(const Snapshot& snapshot) {
    if (!snapshot.output) return;
    const auto& output = *snapshot.output;
    const auto& a = snapshot.identity;
    const auto& b = output.identity;
    if (a.request_epoch != b.request_epoch || a.block_id != b.block_id ||
        a.layer != b.layer || a.start_position != b.start_position)
      throw std::invalid_argument("Cold-KV direct output identity mismatch");
    const auto mul = [](size_t x, size_t y) {
      if (y && x > std::numeric_limits<size_t>::max() / y)
        throw std::invalid_argument("Cold-KV direct output extent overflow");
      return x * y;
    };
    const auto add = [](size_t x, size_t y) {
      if (x > std::numeric_limits<size_t>::max() - y)
        throw std::invalid_argument("Cold-KV direct output extent overflow");
      return x + y;
    };
    struct Range { uintptr_t begin, end; };
    const auto range = [&](const void* pointer, size_t count, size_t width) {
      const size_t bytes = mul(count, width);
      const uintptr_t begin = reinterpret_cast<uintptr_t>(pointer);
      if (!pointer || bytes > std::numeric_limits<uintptr_t>::max() - begin)
        throw std::invalid_argument("Cold-KV direct output null/overflow address");
      return Range{begin, begin + bytes};
    };
    std::array<Range, 6> ranges;
    size_t index = 0;
    for (const QuantizedTarget* target : {&output.keys, &output.values}) {
      if (reinterpret_cast<uintptr_t>(target->data) % alignof(uint32_t) ||
          target->vector_stride_words < kWordsPerVector ||
          target->vector_stride_groups < kGroupsPerVector)
        throw std::invalid_argument("Cold-KV direct output alignment/row stride");
      const size_t word_row_end = add(mul(kTokens - 1, target->vector_stride_words), kWordsPerVector);
      const size_t group_row_end = add(mul(kTokens - 1, target->vector_stride_groups), kGroupsPerVector);
      if (target->head_stride_words < word_row_end || target->head_stride_groups < group_row_end)
        throw std::invalid_argument("Cold-KV direct output overlapping heads");
      const size_t words = add(mul(kHeads - 1, target->head_stride_words), word_row_end);
      const size_t groups = add(mul(kHeads - 1, target->head_stride_groups), group_row_end);
      if (words > target->data_capacity_words || groups > target->scales_capacity_groups ||
          groups > target->biases_capacity_groups)
        throw std::invalid_argument("Cold-KV direct output exceeds capacity");
      ranges[index++] = range(target->data, words, sizeof(uint32_t));
      ranges[index++] = range(target->scales, groups, sizeof(uint16_t));
      ranges[index++] = range(target->biases, groups, sizeof(uint16_t));
    }
    const auto overlaps = [](Range x, Range y) { return x.begin < y.end && y.begin < x.end; };
    for (size_t i = 0; i < ranges.size(); ++i) {
      for (size_t j = 0; j < i; ++j)
        if (overlaps(ranges[i], ranges[j]))
          throw std::invalid_argument("Cold-KV direct output arrays overlap");
      for (const auto* source : {&snapshot.keys, &snapshot.values})
        if (!source->empty() && overlaps(ranges[i], range(source->data(), source->size(), sizeof(uint16_t))))
          throw std::invalid_argument("Cold-KV direct output aliases snapshot input");
    }
  }

  // CPU-only leaf shared by the producer and tests. Exact pinned MLX Metal
  // affine_quantize algorithm for FP16 input, g64/b8: w_max starts at zero,
  // round is nearest with ties away, and codes use FP32 scale before its FP16
  // storage rounding. MLX CPU uses rint and a different all-negative bound;
  // this deliberately matches the GPU algorithm, not CPU-byte equivalence.
  static void packAffineGroup64(const uint16_t* source, uint16_t minimum_bits,
      uint16_t maximum_bits, uint32_t* words, uint16_t& scale_bits,
      uint16_t& bias_bits) {
    if (!source || !words) throw std::invalid_argument("Null affine group buffer");
    const auto from_half = [](uint16_t bits) {
      static_assert(sizeof(_Float16) == sizeof(uint16_t));
      _Float16 value; std::memcpy(&value, &bits, sizeof(bits));
      return static_cast<float>(value);
    };
    const auto to_half = [](float value) {
      const _Float16 half = static_cast<_Float16>(value);
      uint16_t bits; std::memcpy(&bits, &half, sizeof(bits)); return bits;
    };
    const float minimum = from_half(minimum_bits), maximum = from_half(maximum_bits);
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum)
      throw std::runtime_error("Invalid affine group bounds");
    const float w_max = std::max(maximum, 0.0f);
    float scale = std::max((w_max - minimum) / 255.0f, 1e-7f);
    const bool side = std::abs(minimum) > std::abs(w_max);
    scale = side ? scale : -scale;
    const float edge = side ? minimum : w_max;
    const float q0 = std::round(edge / scale);
    const float bias = q0 == 0.0f ? 0.0f : edge;
    if (q0 != 0.0f) scale = edge / q0;
    if (!std::isfinite(scale) || scale == 0.0f)
      throw std::runtime_error("Invalid affine group scale");
    scale_bits = to_half(scale); bias_bits = to_half(bias);
    if (!std::isfinite(from_half(scale_bits)) || from_half(scale_bits) == 0.0f)
      throw std::runtime_error("Affine FP16 scale not representable");
    for (size_t word = 0; word < kGroupSize / 4; ++word) {
      uint32_t packed = 0;
      for (size_t lane = 0; lane < 4; ++lane) {
        const float value = from_half(source[word * 4 + lane]);
        if (!std::isfinite(value) || value < minimum || value > maximum)
          throw std::runtime_error("CoreML affine bounds do not enclose original FP16 group");
        const float code = std::min(255.0f, std::max(0.0f, std::round((value - bias) / scale)));
        packed |= uint32_t(code) << (8 * lane);
      }
      words[word] = packed;
    }
  }

  // Target must have passed validateOutputTarget and remain unchanged. Only
  // raw FP16 bit copies touch scales/biases, so typed half storage is safe.
  static void packAffineGroup64ToTarget(const uint16_t* source,
      uint16_t minimum_bits, uint16_t maximum_bits,
      const QuantizedTarget& target, size_t vector, size_t group) {
    if (vector >= kVectors || group >= kGroupsPerVector)
      throw std::invalid_argument("Cold-KV direct output vector/group index");
    const size_t head = vector / kTokens, token = vector % kTokens;
    const size_t word = head * target.head_stride_words + token * target.vector_stride_words + group * (kGroupSize / 4);
    const size_t metadata = head * target.head_stride_groups + token * target.vector_stride_groups + group;
    uint16_t scale, bias;
    packAffineGroup64(source, minimum_bits, maximum_bits, target.data + word, scale, bias);
    std::memcpy(static_cast<uint8_t*>(target.scales) + metadata * sizeof(scale), &scale, sizeof(scale));
    std::memcpy(static_cast<uint8_t*>(target.biases) + metadata * sizeof(bias), &bias, sizeof(bias));
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
