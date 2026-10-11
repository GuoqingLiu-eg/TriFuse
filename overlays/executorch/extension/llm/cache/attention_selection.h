/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace executorch::extension::llm::cache {

struct AttentionSelection {
  uint64_t request_epoch = 0;
  uint64_t proposal_id = 0;
  int covered_length = 0;
  int valid_until_position = 0;
  std::vector<std::vector<int32_t>> positions_by_head;
};

// Contains advice only; it never reads or modifies cache storage. The request
// epoch is separate from any owner's per-token generation counter.
class AttentionSelectionMailbox {
 public:
  AttentionSelectionMailbox(int capacity, int heads)
      : capacity_(capacity), heads_(heads) {
    if (capacity <= 0 || heads <= 0) {
      throw std::invalid_argument("attention selection: positive capacity/heads required");
    }
  }

  uint64_t begin_request() {
    std::shared_ptr<const AttentionSelection> retired;
    uint64_t next;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (epoch_ == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("attention selection: request epoch exhausted");
      }
      next = ++epoch_;
      accepted_id_ = 0;
      has_accepted_id_ = false;
      latest_.swap(retired);
    }
    return next;
  }

  uint64_t epoch() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return epoch_;
  }

  // The caller must not mutate its proposal during this call. Subsequent
  // mutations through another alias cannot change the published deep copy.
  bool publish(std::shared_ptr<const AttentionSelection> proposal) {
    if (!proposal) {
      return false;
    }
    std::shared_ptr<const AttentionSelection> frozen =
        std::make_shared<AttentionSelection>(*proposal);
    if (!valid(*frozen)) {
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (epoch_ == 0 || frozen->request_epoch != epoch_ ||
          (has_accepted_id_ && frozen->proposal_id <= accepted_id_)) {
        return false;
      }
      accepted_id_ = frozen->proposal_id;
      has_accepted_id_ = true;
      latest_.swap(frozen);
    }
    // The replaced proposal, including its index vectors, is released unlocked.
    return true;
  }

  // A busy publisher never stalls the inference thread. Advice is reusable;
  // future coverage remains pending and an inclusive deadline ends its use.
  std::shared_ptr<const AttentionSelection> take_latest(int current_position) {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || current_position < 0 ||
        current_position > capacity_ || !latest_ ||
        latest_->request_epoch != epoch_ ||
        latest_->covered_length > current_position ||
        current_position > latest_->valid_until_position) {
      return nullptr;
    }
    return latest_;
  }

 private:
  bool valid(const AttentionSelection& proposal) const {
    if (proposal.covered_length < 1 || proposal.covered_length > capacity_ ||
        proposal.valid_until_position < proposal.covered_length ||
        proposal.valid_until_position > capacity_ ||
        proposal.positions_by_head.size() != static_cast<size_t>(heads_)) {
      return false;
    }
    const size_t count = proposal.positions_by_head.front().size();
    if (count == 0 || count > static_cast<size_t>(capacity_)) {
      return false;
    }
    for (const auto& positions : proposal.positions_by_head) {
      if (positions.size() != count) {
        return false;
      }
      int32_t previous = -1;
      for (int32_t position : positions) {
        if (position <= previous || position >= proposal.covered_length) {
          return false;
        }
        previous = position;
      }
    }
    return true;
  }

  const int capacity_;
  const int heads_;
  mutable std::mutex mutex_;
  uint64_t epoch_ = 0;
  uint64_t accepted_id_ = 0;
  bool has_accepted_id_ = false;
  std::shared_ptr<const AttentionSelection> latest_;
};

} // namespace executorch::extension::llm::cache
