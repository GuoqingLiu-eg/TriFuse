#pragma once

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace trifuse_cold_kv_experiment {

// Owner-thread policy only: no allocation, clocks, request IDs, cache mutation,
// or hard memory-limit promise. The caller serializes evaluate -> conversion ->
// published. A decision is not a reservation and must not be queued/replayed.
class ThresholdPolicy {
 public:
  static constexpr uint64_t kMiB = uint64_t{1024} * 1024;
  struct Config {
    uint64_t high_bytes = 448 * kMiB;
    uint64_t min_growth_bytes = 32 * kMiB;
    bool adaptive = true;
    uint64_t min_high_bytes = 256 * kMiB;
    uint64_t max_high_bytes = 512 * kMiB;
    uint64_t adjustment_step_bytes = 32 * kMiB;
  };
  struct Geometry {
    int64_t layers = 16;
    int64_t heads = 16;
    int64_t dim = 128;
  };
  enum class Reason {
    AtOrBelowHigh,
    NoNewEligibleBlock,
    GrowthNotMet,
    InitialHigh,
    RearmedHigh,
    GrowthAfterPublish,
  };
  struct Decision {
    bool trigger = false;
    Reason reason = Reason::AtOrBelowHigh;
    int64_t from_token = 0; // Already cold: never encode tokens before here.
    int64_t to_token = 0;   // Exclusive end of all eligible complete old blocks.
    uint64_t new_blocks = 0;
    uint64_t fp16_bytes_to_encode = 0;
    // Logical packed codes + FP16 scale/bias per group64, NOT an allocation prediction.
    uint64_t encoded_payload_bytes = 0;
  };
  struct Stats {
    uint64_t evaluations = 0;
    uint64_t triggers = 0;
    uint64_t publications = 0;
    uint64_t cache_resets = 0;
    uint64_t invalid_inputs = 0;
    uint64_t skip_at_or_below_high = 0;
    uint64_t skip_no_new_block = 0;
    uint64_t skip_growth_not_met = 0;
    uint64_t trigger_initial = 0;
    uint64_t trigger_rearmed = 0;
    uint64_t trigger_growth = 0;
    uint64_t rearm_observations = 0;
    uint64_t threshold_increases = 0;
    uint64_t threshold_increases_capped = 0;
    uint64_t memory_ceiling_updates = 0;
  };
  struct State {
    bool has_publication = false;
    bool rearmed = true;
    uint64_t last_published_bytes = 0;
    int64_t last_published_cold_tokens = 0;
    uint64_t target_high_bytes = 0;
    bool memory_ceiling_enabled = false;
    uint64_t memory_ceiling_bytes = 0;
    bool ceiling_floor_limited = false;
    bool last_cost_available = false;
    uint64_t last_before_bytes = 0;
    double last_transaction_ms = 0;
    double last_reclaimed_fraction = 0;
  };

  ThresholdPolicy() : ThresholdPolicy(Config{}, Geometry{}, 512, 256) {}
  explicit ThresholdPolicy(Config config)
      : ThresholdPolicy(config, Geometry{}, 512, 256) {}
  ThresholdPolicy(Config config, Geometry geometry,
                  int64_t hot_recent = 512, int64_t block = 256)
      : config_(config), hot_recent_(hot_recent), block_(block) {
    if (!config.high_bytes || !config.min_growth_bytes || !config.min_high_bytes ||
        config.min_high_bytes > config.high_bytes || config.high_bytes > config.max_high_bytes ||
        !config.adjustment_step_bytes || hot_recent < 0 ||
        block <= 0 || geometry.layers <= 0 || geometry.heads <= 0 || geometry.dim <= 0 || geometry.dim % 64 != 0)
      throw std::invalid_argument("ThresholdPolicy: invalid limits or geometry");
    const auto vectors = multiply(uint64_t(geometry.layers), uint64_t(geometry.heads));
    fp16_bytes_per_token_ = multiply(multiply(vectors, uint64_t(geometry.dim)), 4);
    // K and V each have dim packed bytes plus FP16 scale+bias per group64.
    encoded_bytes_per_token_ = multiply(multiply(vectors, 2),
        add(uint64_t(geometry.dim), multiply(uint64_t(geometry.dim / 64), 4)));
    state_.target_high_bytes = config.high_bytes;
  }

  Decision evaluate(uint64_t current_bytes, int64_t logical_length,
                    int64_t already_cold_tokens) {
    increment(stats_.evaluations);
    Decision result;
    try {
      if (logical_length < 0 || already_cold_tokens < 0 ||
          already_cold_tokens > logical_length || already_cold_tokens % block_ != 0 ||
          (state_.has_publication &&
           already_cold_tokens < state_.last_published_cold_tokens))
        throw std::invalid_argument("ThresholdPolicy: invalid positions; reset after cache clear");
      result.from_token = result.to_token = already_cold_tokens;
      if (current_bytes <= effectiveHighBytes()) {
        result.reason = Reason::AtOrBelowHigh;
      } else {
        // Subtraction and multiplication remain within nonnegative int64_t.
        const int64_t eligible = logical_length > hot_recent_
            ? ((logical_length - hot_recent_) / block_) * block_ : 0;
        if (eligible <= already_cold_tokens) {
          result.reason = Reason::NoNewEligibleBlock;
        } else if (state_.has_publication && !state_.rearmed &&
                   (current_bytes < state_.last_published_bytes ||
                    current_bytes - state_.last_published_bytes < config_.min_growth_bytes)) {
          result.reason = Reason::GrowthNotMet;
        } else {
          result.trigger = true;
          result.reason = !state_.has_publication ? Reason::InitialHigh :
              state_.rearmed ? Reason::RearmedHigh : Reason::GrowthAfterPublish;
          result.to_token = eligible;
          const auto tokens = uint64_t(eligible - already_cold_tokens);
          result.new_blocks = tokens / uint64_t(block_);
          result.fp16_bytes_to_encode = multiply(tokens, fp16_bytes_per_token_);
          result.encoded_payload_bytes = multiply(tokens, encoded_bytes_per_token_);
        }
      }
    } catch (...) {
      last_decision_ = {}; // An invalid observation cannot authorize publication.
      increment(stats_.invalid_inputs);
      throw;
    }
    // Equality is neither a trigger nor a below-high rearm observation.
    if (current_bytes < effectiveHighBytes() && !state_.rearmed) {
      state_.rearmed = true;
      increment(stats_.rearm_observations);
    }
    switch (result.reason) {
      case Reason::AtOrBelowHigh: increment(stats_.skip_at_or_below_high); break;
      case Reason::NoNewEligibleBlock: increment(stats_.skip_no_new_block); break;
      case Reason::GrowthNotMet: increment(stats_.skip_growth_not_met); break;
      case Reason::InitialHigh: increment(stats_.trigger_initial); break;
      case Reason::RearmedHigh: increment(stats_.trigger_rearmed); break;
      case Reason::GrowthAfterPublish: increment(stats_.trigger_growth); break;
    }
    if (result.trigger) increment(stats_.triggers);
    last_decision_ = result;
    return result;
  }

  // Call only after a successful, synchronous cache publication. Failure to
  // reach below high is valid: future actions require actual allocation growth.
  void published(uint64_t after_bytes, int64_t new_cold_tokens) {
    publish(after_bytes, new_cold_tokens, false, 0, 0);
  }

  // Measured conversion cost only; no request length or wall-clock sampling.
  // Slow (>=100 ms) or ineffective (<20% reclaimed) conversion raises the next
  // target by one bounded step. Good results never cause automatic lowering.
  void published(uint64_t after_bytes, int64_t new_cold_tokens,
                 uint64_t before_bytes, double elapsed_ms) {
    if (!before_bytes || !std::isfinite(elapsed_ms) || elapsed_ms < 0)
      throw std::invalid_argument("ThresholdPolicy: invalid publication cost");
    publish(after_bytes, new_cold_tokens, true, before_bytes, elapsed_ms);
  }

  // Explicit externally supplied KV budget. It is not system-memory detection
  // or a hard cap: a ceiling below configured min is floor-limited and reported.
  void set_memory_ceiling(uint64_t bytes) {
    if (!bytes) throw std::invalid_argument("ThresholdPolicy: zero memory ceiling");
    increment(stats_.memory_ceiling_updates);
    state_.memory_ceiling_enabled = true;
    state_.memory_ceiling_bytes = bytes;
    state_.ceiling_floor_limited = bytes < config_.min_high_bytes;
  }
  void clear_memory_ceiling() {
    increment(stats_.memory_ceiling_updates);
    state_.memory_ceiling_enabled = false;
    state_.memory_ceiling_bytes = 0;
    state_.ceiling_floor_limited = false;
  }
  uint64_t targetHighBytes() const noexcept { return state_.target_high_bytes; }
  uint64_t effectiveHighBytes() const noexcept {
    return state_.memory_ceiling_enabled
        ? std::max(config_.min_high_bytes,
                   std::min(state_.target_high_bytes, state_.memory_ceiling_bytes))
        : state_.target_high_bytes;
  }

  // Explicit CACHE lifecycle reset; request boundaries alone do not call this.
  // Keep counters and externally supplied budget; reset the adaptive target.
  void reset() {
    increment(stats_.cache_resets);
    const bool enabled = state_.memory_ceiling_enabled;
    const auto ceiling = state_.memory_ceiling_bytes;
    state_ = {};
    state_.target_high_bytes = config_.high_bytes;
    state_.memory_ceiling_enabled = enabled;
    state_.memory_ceiling_bytes = ceiling;
    state_.ceiling_floor_limited = enabled && ceiling < config_.min_high_bytes;
    last_decision_ = {};
  }

  const Stats& stats() const noexcept { return stats_; }
  const State& state() const noexcept { return state_; }
  const Config& config() const noexcept { return config_; }
  static const char* reasonName(Reason reason) noexcept {
    switch (reason) {
      case Reason::AtOrBelowHigh: return "at_or_below_high";
      case Reason::NoNewEligibleBlock: return "no_new_eligible_block";
      case Reason::GrowthNotMet: return "allocation_growth_not_met";
      case Reason::InitialHigh: return "initial_high_exceeded";
      case Reason::RearmedHigh: return "rearmed_high_exceeded";
      case Reason::GrowthAfterPublish: return "growth_after_publication";
    }
    return "invalid";
  }

 private:
  void publish(uint64_t after_bytes, int64_t new_cold_tokens, bool cost_available,
               uint64_t before_bytes, double elapsed_ms) {
    if (!last_decision_.trigger || new_cold_tokens != last_decision_.to_token ||
        new_cold_tokens <= last_decision_.from_token)
      throw std::invalid_argument("ThresholdPolicy: publication does not match last decision");
    const double reclaimed = cost_available
        ? (double(before_bytes) - double(after_bytes)) / double(before_bytes) : 0;
    if (config_.adaptive && cost_available && (elapsed_ms >= 100.0 || reclaimed < 0.2)) {
      const auto room = config_.max_high_bytes - state_.target_high_bytes;
      const auto increase = std::min(room, config_.adjustment_step_bytes);
      if (increase) {
        state_.target_high_bytes += increase;
        increment(stats_.threshold_increases);
      } else {
        increment(stats_.threshold_increases_capped);
      }
    }
    increment(stats_.publications);
    state_.has_publication = true;
    state_.rearmed = after_bytes < effectiveHighBytes();
    state_.last_published_bytes = after_bytes;
    state_.last_published_cold_tokens = new_cold_tokens;
    state_.last_cost_available = cost_available;
    state_.last_before_bytes = before_bytes;
    state_.last_transaction_ms = elapsed_ms;
    state_.last_reclaimed_fraction = reclaimed;
    last_decision_ = {};
  }
  static uint64_t multiply(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
      throw std::overflow_error("ThresholdPolicy: logical payload overflow");
    return a * b;
  }
  static uint64_t add(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b)
      throw std::overflow_error("ThresholdPolicy: byte count overflow");
    return a + b;
  }
  static void increment(uint64_t& value) { value = add(value, 1); }
  Config config_;
  int64_t hot_recent_, block_;
  uint64_t fp16_bytes_per_token_ = 0, encoded_bytes_per_token_ = 0;
  Stats stats_;
  State state_;
  Decision last_decision_;
};

} // namespace trifuse_cold_kv_experiment
