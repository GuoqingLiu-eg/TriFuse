/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Licensed under the BSD-style license in the repository LICENSE file.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace executorch::extension::llm::cache {

// Tensor-free address layout. Absolute lengths remain owned by the sequence
// controller/caller; stored rows are a sparse old prefix followed by a dense
// newly appended tail. This class does not change RoPE positions, select a
// retention policy, or extend the original model's absolute capacity.
class RetainedSequenceLayout {
 public:
  struct AppendPlan {
    int physical_start;
    int physical_end;
    int absolute_end;
  };

  class CompactionPlan {
   public:
    const std::vector<int32_t>& source_slots() const noexcept {
      return source_slots_;
    }
    const std::vector<int32_t>& kept_positions() const noexcept {
      return kept_positions_;
    }
    int absolute_length() const noexcept {
      return absolute_length_;
    }
    int retained_length() const noexcept {
      return static_cast<int>(source_slots_.size());
    }
    int max_physical_slots() const noexcept {
      return max_physical_slots_;
    }

   private:
    friend class RetainedSequenceLayout;
    CompactionPlan(
        const RetainedSequenceLayout* owner,
        uint64_t reset_epoch,
        uint64_t revision,
        int length,
        int maximum,
        std::vector<int32_t> kept,
        std::vector<int32_t> slots)
        : owner_(owner),
          reset_epoch_(reset_epoch),
          revision_(revision),
          absolute_length_(length),
          max_physical_slots_(maximum),
          kept_positions_(std::move(kept)),
          source_slots_(std::move(slots)) {}

    const RetainedSequenceLayout* owner_;
    uint64_t reset_epoch_;
    uint64_t revision_;
    int absolute_length_;
    int max_physical_slots_;
    std::vector<int32_t> kept_positions_;
    std::vector<int32_t> source_slots_;
  };

  explicit RetainedSequenceLayout(int capacity) : capacity_(capacity) {
    if (capacity <= 0 || static_cast<int64_t>(capacity) > INT32_MAX) {
      throw std::invalid_argument("retained layout: invalid absolute capacity");
    }
  }

  bool compacted() const noexcept {
    return compacted_;
  }
  int compacted_at_abs_length() const noexcept {
    return anchor_;
  }
  const std::vector<int32_t>& retained_prefix() const noexcept {
    return prefix_;
  }
  int max_physical_slots() const noexcept {
    return capacity_ - anchor_ + static_cast<int>(prefix_.size());
  }
  int retained_length(int layer_abs_length) const {
    check_length(layer_abs_length);
    return layer_abs_length - anchor_ + static_cast<int>(prefix_.size());
  }

  // A missing original position is not a reusable old token. It can never be
  // recovered by interpreting a current physical slot as its old absolute ID.
  std::optional<int32_t> source_slot(
      int original_position,
      int layer_abs_length) const {
    check_length(layer_abs_length);
    if (original_position < 0 || original_position >= layer_abs_length) {
      return std::nullopt;
    }
    if (original_position >= anchor_) {
      return static_cast<int32_t>(
          prefix_.size() + static_cast<size_t>(original_position - anchor_));
    }
    const auto found = std::lower_bound(
        prefix_.begin(), prefix_.end(), original_position);
    if (found == prefix_.end() || *found != original_position) {
      return std::nullopt;
    }
    return static_cast<int32_t>(found - prefix_.begin());
  }

  AppendPlan plan_append(int layer_abs_length, int absolute_start, int T) const {
    check_length(layer_abs_length);
    if (absolute_start != layer_abs_length || T <= 0 ||
        T > capacity_ - absolute_start) {
      throw std::invalid_argument("retained layout: invalid absolute append");
    }
    const int physical_start = retained_length(layer_abs_length);
    return {physical_start, physical_start + T, absolute_start + T};
  }

  // All layers must have completed the same absolute step. Candidate metadata
  // is private and allocated before any GPU publication; this call is pure.
  CompactionPlan prepare_compaction(
      const std::vector<int32_t>& keep_original_positions,
      int expected_abs_length,
      const std::vector<int>& layer_abs_lengths) const {
    check_length(expected_abs_length);
    if (!same_length(layer_abs_lengths, expected_abs_length) ||
        keep_original_positions.empty() || revision_ == UINT64_MAX) {
      throw std::invalid_argument("retained layout: incomplete layers/empty keep set");
    }
    std::vector<int32_t> slots;
    slots.reserve(keep_original_positions.size());
    int32_t previous = -1;
    for (const int32_t position : keep_original_positions) {
      if (position <= previous) {
        throw std::invalid_argument("retained layout: keep positions must increase");
      }
      const auto slot = source_slot(position, expected_abs_length);
      if (!slot) {
        throw std::invalid_argument("retained layout: keep position is unavailable");
      }
      slots.push_back(*slot);
      previous = position;
    }
    const int maximum = capacity_ - expected_abs_length + static_cast<int>(slots.size());
    return CompactionPlan(this, reset_epoch_, revision_, expected_abs_length,
        maximum, keep_original_positions, std::move(slots));
  }

  bool can_publish(
      const CompactionPlan& plan,
      const std::vector<int>& layer_abs_lengths) const noexcept {
    return plan.owner_ == this && plan.reset_epoch_ == reset_epoch_ &&
        plan.revision_ == revision_ && revision_ != UINT64_MAX &&
        same_length(layer_abs_lengths, plan.absolute_length_);
  }

  // Caller serializes evaluation, can_publish, all-pool swaps and this commit.
  // No allocation or throwing work occurs after swapping tensor ownership.
  // The plan is consumed; inspect its kept_positions before publication.
  bool publish(
      CompactionPlan&& plan,
      const std::vector<int>& layer_abs_lengths) noexcept {
    if (!can_publish(plan, layer_abs_lengths)) {
      return false;
    }
    prefix_.swap(plan.kept_positions_);
    anchor_ = plan.absolute_length_;
    compacted_ = true;
    ++revision_;
    return true;
  }

  void reset() noexcept {
    prefix_.clear();
    anchor_ = 0;
    revision_ = 0;
    compacted_ = false;
    ++reset_epoch_; // Invalidates plans across clear/re-prefill (same-length ABA).
  }

 private:
  void check_length(int length) const {
    if (length < anchor_ || length > capacity_) {
      throw std::invalid_argument("retained layout: invalid absolute layer length");
    }
  }
  static bool same_length(const std::vector<int>& lengths, int expected) noexcept {
    return !lengths.empty() && std::all_of(lengths.begin(), lengths.end(),
        [expected](int length) { return length == expected; });
  }

  int capacity_;
  int anchor_ = 0;
  std::vector<int32_t> prefix_;
  uint64_t reset_epoch_ = 0;
  uint64_t revision_ = 0;
  bool compacted_ = false;
};

} // namespace executorch::extension::llm::cache
