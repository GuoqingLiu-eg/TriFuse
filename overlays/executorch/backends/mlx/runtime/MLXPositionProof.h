// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
#pragma once

#include "MLXLoader.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <unordered_map>

namespace executorch::backends::mlx {

// A narrow, immutable proof about the main chain, not an override of positions.
// Cache it with the exact loaded program; bind fresh input bytes every execute.
struct InputPositionProof {
  size_t input_index{1};
  uint32_t input_tid{0};
  ScalarType scalar_type{ScalarType::Long};
  std::vector<int32_t> shape;
  size_t count{0};
  size_t update_count{0};
};

namespace position_proof_detail {
using Shape = std::vector<int32_t>;

template <class T, class = void> struct HasOut : std::false_type {};
template <class T> struct HasOut<T, std::void_t<decltype(std::declval<T>().out)>>
    : std::true_type {};
template <class T, class = void> struct HasOuts : std::false_type {};
template <class T> struct HasOuts<T, std::void_t<decltype(std::declval<T>().outs)>>
    : std::true_type {};
template <class T, class = void> struct HasOutputs : std::false_type {};
template <class T> struct HasOutputs<T, std::void_t<decltype(std::declval<T>().outputs)>>
    : std::true_type {};

inline std::optional<size_t> elements(const Shape& shape) {
  size_t size = 1;
  for (int32_t dim : shape) {
    if (dim <= 0 || size > size_t(INT32_MAX) / size_t(dim)) return std::nullopt;
    size *= size_t(dim);
  }
  return size;
}

inline std::optional<size_t> axis(int32_t value, size_t rank) {
  const int64_t normalized = value < 0 ? int64_t(value) + int64_t(rank) : value;
  if (normalized < 0 || normalized >= int64_t(rank)) return std::nullopt;
  return size_t(normalized);
}

inline std::optional<Shape> reshape(
    const Shape& input, const std::vector<std::variant<int64_t, Vid>>& dims) {
  const auto count = elements(input);
  if (!count) return std::nullopt;
  Shape output;
  size_t product = 1;
  std::optional<size_t> inferred;
  for (const auto& dim : dims) {
    const auto* value = std::get_if<int64_t>(&dim);
    if (!value) return std::nullopt; // Dynamic shape values are not proven.
    if (*value == -1 && !inferred) {
      inferred = output.size();
      output.push_back(-1);
    } else {
      if (*value <= 0 || *value > INT32_MAX || product > *count / size_t(*value))
        return std::nullopt;
      product *= size_t(*value);
      output.push_back(int32_t(*value));
    }
  }
  if (inferred) {
    if (*count % product) return std::nullopt;
    output[*inferred] = int32_t(*count / product);
  } else if (product != *count) {
    return std::nullopt;
  }
  return output;
}
} // namespace position_proof_detail

// Only linear-order-preserving reshape / expand / squeeze / singleton scalar
// take are recognized. Unknown positions fall back to normal MLX evaluation.
// All writes invalidate their output Tid first: serialized slots are reused.
inline std::optional<InputPositionProof> prove_input_positions(
    const MLXProgram& program) {
  using namespace position_proof_detail;
  if (program.input_map.size() <= 1 ||
      program.input_map[1].slot_type != SlotType::TensorSlot ||
      program.main_chain_idx >= program.instruction_chains.size()) return std::nullopt;
  const auto tid = program.input_map[1].idx;
  if (tid >= program.tensor_meta.size() || !program.tensor_meta[tid]) return std::nullopt;
  const auto& meta = *program.tensor_meta[tid];
  if (meta.scalar_type != ScalarType::Int && meta.scalar_type != ScalarType::Long)
    return std::nullopt;
  InputPositionProof proof;
  proof.input_tid = tid;
  proof.scalar_type = meta.scalar_type;
  for (const auto& d : meta.shape) {
    if (d.is_dynamic() || d.value <= 0) return std::nullopt;
    proof.shape.push_back(d.value);
  }
  const auto count = elements(proof.shape);
  if (!count) return std::nullopt;
  proof.count = *count;
  std::unordered_map<uint32_t, Shape> known{{tid, proof.shape}};
  try {
    for (const auto& instruction : program.instruction_chains[program.main_chain_idx]) {
      // Nested control flow or custom kernels could hide writes or positions.
      if (instruction.op == OpCode::SCAN || instruction.op == OpCode::IF ||
          instruction.op == OpCode::METAL_KERNEL) return std::nullopt;
      std::optional<std::pair<uint32_t, Shape>> propagated;
      auto input_shape = [&](Tid input) -> const Shape* {
        const auto it = known.find(input.idx);
        return it == known.end() ? nullptr : &it->second;
      };
      switch (instruction.op) {
        case OpCode::EXPAND_DIMS: {
          const auto& n = std::get<ExpandDimsNode>(instruction.node);
          if (const auto* input = input_shape(n.x)) {
            if (const auto ax = axis(n.axis, input->size() + 1)) {
              Shape output = *input;
              output.insert(output.begin() + *ax, 1);
              propagated = {{n.out.idx, std::move(output)}};
            }
          }
          break;
        }
        case OpCode::RESHAPE: {
          const auto& n = std::get<ReshapeNode>(instruction.node);
          if (const auto* input = input_shape(n.x)) {
            if (auto output = reshape(*input, n.shape))
              propagated = {{n.out.idx, std::move(*output)}};
          }
          break;
        }
        case OpCode::SQUEEZE: {
          const auto& n = std::get<SqueezeNode>(instruction.node);
          if (const auto* input = input_shape(n.x)) {
            Shape output;
            std::vector<bool> remove(input->size(), false);
            bool valid = true;
            if (n.dims.empty()) {
              for (size_t i = 0; i < input->size(); ++i) remove[i] = (*input)[i] == 1;
            } else {
              for (int32_t dim : n.dims) {
                const auto ax = axis(dim, input->size());
                if (!ax || remove[*ax]) { valid = false; break; }
                remove[*ax] = (*input)[*ax] == 1;
              }
            }
            if (valid) {
              for (size_t i = 0; i < input->size(); ++i)
                if (!remove[i]) output.push_back((*input)[i]);
              propagated = {{n.out.idx, std::move(output)}};
            }
          }
          break;
        }
        case OpCode::TAKE: {
          const auto& n = std::get<TakeNode>(instruction.node);
          if (const auto* input = input_shape(n.x)) {
            const auto ax = axis(n.axis, input->size());
            if (ax && (*input)[*ax] == 1 && n.index.kind == 0 &&
                (n.index.literal == 0 || n.index.literal == -1)) {
              Shape output = *input;
              output.erase(output.begin() + *ax);
              propagated = {{n.out.idx, std::move(output)}};
            }
          }
          break;
        }
        case OpCode::UPDATE_AND_ATTEND: {
          const auto& n = std::get<UpdateAndAttendNode>(instruction.node);
          if (!input_shape(n.position)) return std::nullopt;
          ++proof.update_count;
          break;
        }
        default: break;
      }
      // Read proof before invalidating, so x==out is handled correctly.
      std::visit([&](const auto& n) {
        using N = std::decay_t<decltype(n)>;
        if constexpr (HasOut<N>::value) {
          if constexpr (std::is_same_v<std::decay_t<decltype(n.out)>, Tid>)
            known.erase(n.out.idx);
        }
        if constexpr (HasOuts<N>::value)
          for (const auto& output : n.outs) known.erase(output.idx);
        if constexpr (HasOutputs<N>::value)
          for (const auto& output : n.outputs) known.erase(output.idx);
      }, instruction.node);
      if (propagated) known.insert_or_assign(propagated->first, std::move(propagated->second));
    }
  } catch (const std::bad_variant_access&) {
    return std::nullopt;
  }
  if (!proof.update_count) return std::nullopt;
  return proof;
}

// Backend must also validate the live input shape, dtype and contiguity against
// proof and its normal Tensor ABI before calling this. Copy actual input bytes;
// never substitute cache.length(), a caller's claimed position, or prior calls.
template <typename Integer>
inline std::vector<int32_t> copy_checked_host_positions(
    const InputPositionProof& proof, const Integer* data, size_t count) {
  static_assert(std::is_same_v<Integer, int32_t> || std::is_same_v<Integer, int64_t>);
  const auto dtype = std::is_same_v<Integer, int32_t> ? ScalarType::Int : ScalarType::Long;
  if (!data || count != proof.count || dtype != proof.scalar_type)
    throw std::invalid_argument("host positions: input ABI differs from proof");
  std::vector<int32_t> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const int64_t value = data[i];
    if (value < 0 || value > INT32_MAX)
      throw std::out_of_range("host positions: value outside nonnegative int32 range");
    result.push_back(int32_t(value));
  }
  return result;
}
} // namespace executorch::backends::mlx
