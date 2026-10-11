// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
#pragma once

#include "MLXPositionProof.h"
#include "MLXOLMoERouter.h"
#include "MLXMoEExpertHook.h"
#include "MLXMoEWeightPrefetch.h"
#include "MLXSequenceCache.h"
#include <trifuse/kv/ColdKVAttention.h>
#include "resident_prefill.h"
#include "resident_features.h"
#include "resident_attention_labels.h"
#include "resident_greedy.h"
#include <mlx/compile.h>
#include <mlx/fast.h>
#include <mlx/utils.h>
#include <mlx/version.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

namespace executorch::backends::mlx {

// Diagnostic A/B selector: compare operators in the same signed phone binary
// with the same scheduling/cache policy. Keep the established path as default
// until complete-request measurements justify promoting a candidate.
enum class NativeMoEOperatorMode { Baseline, Inverse, Router };
inline NativeMoEOperatorMode native_moe_operator_mode() {
  const char* value = std::getenv("TRIFUSE_MOE_OPERATOR_MODE");
  if (!value || std::string(value) == "baseline") return NativeMoEOperatorMode::Baseline;
  if (std::string(value) == "inverse") return NativeMoEOperatorMode::Inverse;
  if (std::string(value) == "router") return NativeMoEOperatorMode::Router;
  throw std::invalid_argument("Unknown TRIFUSE_MOE_OPERATOR_MODE");
}
inline const char* native_moe_operator_mode_name(NativeMoEOperatorMode mode) {
  switch (mode) {
    case NativeMoEOperatorMode::Baseline: return "baseline";
    case NativeMoEOperatorMode::Inverse: return "inverse";
    case NativeMoEOperatorMode::Router: return "router";
  }
  throw std::invalid_argument("Invalid native MoE operator mode");
}

// Kept separate from the adapter so focused fixtures can exercise this exact
// kernel source. K <= 8 matches the ordered small-column reduction used here;
// FP32 products/sums and both intermediate FP16 roundings remain explicit.
inline constexpr const char* kGraniteWeightedResidualSource = R"TRIFUSE_METAL(
#pragma clang fp reassociate(off) contract(off)
const uint out_idx = thread_position_in_grid.x;
const uint row = out_idx / H;
const uint column = out_idx % H;
float total = 0.0f;
for (uint k = 0; k < K; ++k) {
    const float product = float(expert_output[(row * K + k) * H + column]) * gates[row * K + k];
    // col_reduce_small first adds each lane's product to +0, then combines
    // lanes in expert order. A size-one reduction is the identity in MLX.
    const float partial = K == 1 ? product : product + 0.0f;
    total = k == 0 ? partial : partial + total;
}
const half reduced = half(total);
const half scaled = half(reduced * residual_scale);
result[out_idx] = half(residual[out_idx] + scaled);
)TRIFUSE_METAL";

// Equivalent to the pinned Metal argpartition's stable ascending sort and
// final-K slice. NaNs sort last; equal values retain increasing expert IDs.
// The FP32 normalization follows Apple MLX 0.32.3 softmax_single_row's
// four-values-per-lane order, including both SIMD reduction stages.
// Exposed as one source constant so the operator fixture uses this exact code.
inline constexpr const char* kGraniteRouterSource = R"TRIFUSE_ROUTER(
#pragma clang fp reassociate(off) contract(off)
const uint row = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
threadgroup float values[64];
threadgroup float selected_values[8];
threadgroup float local_max[32];
threadgroup float local_normalizer[32];
for (uint i = lane; i < E; i += 32) {
    values[i] = float(router_logits[row * E + i]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Every valid expert gets a unique stable rank; no atomic writes or padded
// expert IDs participate. Read only E real entries, including for E == 40.
for (uint i = lane; i < E; i += 32) {
    const float value = values[i];
    const bool value_nan = metal::isnan(value);
    uint rank = 0;
    for (uint j = 0; j < E; ++j) {
        const float other = values[j];
        const bool other_nan = metal::isnan(other);
        const bool less = !other_nan && (value_nan || other < value);
        const bool equal = (other_nan && value_nan) ||
                           (!other_nan && !value_nan && other == value);
        rank += uint(less || (equal && j < i));
    }
    if (rank >= E - K) {
        const uint slot = rank - (E - K);
        selected_values[slot] = value;
        indices[row * K + slot] = i;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float ld[4];
for (uint i = 0; i < 4; ++i) {
    const uint index = lane * 4 + i;
    ld[i] = index < K ? selected_values[index] : -metal::numeric_limits<float>::infinity();
}
local_max[lane] = -metal::numeric_limits<float>::infinity();
local_normalizer[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);

float maxval = -metal::numeric_limits<float>::max();
for (uint i = 0; i < 4; ++i) {
    maxval = maxval < ld[i] ? ld[i] : maxval;
}
maxval = metal::simd_max(maxval);
if (lane == 0) local_max[0] = maxval;
threadgroup_barrier(mem_flags::mem_threadgroup);
maxval = metal::simd_max(local_max[lane]);
if (lane == 0) local_max[0] = maxval;
threadgroup_barrier(mem_flags::mem_threadgroup);
maxval = local_max[0];

float normalizer = 0.0f;
for (uint i = 0; i < 4; ++i) {
    const float exp_x = metal::fast::exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
}
normalizer = metal::simd_sum(normalizer);
if (lane == 0) local_normalizer[0] = normalizer;
threadgroup_barrier(mem_flags::mem_threadgroup);
normalizer = metal::simd_sum(local_normalizer[lane]);
if (lane == 0) local_normalizer[0] = normalizer;
threadgroup_barrier(mem_flags::mem_threadgroup);
normalizer = 1.0f / local_normalizer[0];
for (uint i = 0; i < 4; ++i) {
    const uint index = lane * 4 + i;
    if (index < K) gates[row * K + index] = ld[i] * normalizer;
}
)TRIFUSE_ROUTER";

// Standard full-attention Granite MoE, using the original MLX affine Q/S/B
// arrays. Expert weights stay packed: only selected embedding rows are
// dequantized; selected experts execute directly through gather_qmm.
class NativeGraniteMoEPrefill {
 public:
  NativeGraniteMoEPrefill(const MLXProgram& program, const ConstantData& constants,
                        NativeModelConfig config, const ::mlx::core::Stream& stream)
      : config_(config), operator_mode_(native_moe_operator_mode()),
        weight_prefetch_mode_(native_moe_weight_prefetch_mode()),
        weight_prefetch_layer_budget_(native_moe_weight_prefetch_layer_budget(weight_prefetch_mode_)),
        weight_prefetch_limits_(native_moe_weight_prefetch_limits(weight_prefetch_mode_)),
        weight_prefetch_scope_(native_moe_weight_prefetch_scope(weight_prefetch_mode_)) {
    using namespace ::mlx::core;
    validate_config(config_);
    std::fprintf(stderr, "TRIFUSE_MOE_OPERATORS mode=%s\n", native_moe_operator_mode_name(operator_mode_));
    need(stream.device.type == Device::gpu, "GPU stream required");
    need(program.init_chain_idx == -1 && program.mutable_buffer_map.empty() &&
         program.input_map.size() == 2 && program.output_map.size() == 1 &&
         program.input_map[0].slot_type == SlotType::TensorSlot &&
         program.input_map[1].slot_type == SlotType::TensorSlot &&
         program.output_map[0].slot_type == SlotType::TensorSlot &&
         program.main_chain_idx >= 0 &&
         size_t(program.main_chain_idx) < program.instruction_chains.size(),
         "unsupported program boundary/init/mutable state");
    const auto proof = prove_input_positions(program);
    need(proof && proof->update_count == size_t(config_.layers), "unproved position/cache wiring");
    const auto& token = metadata(program, Tid{program.input_map[0].idx});
    need(token.scalar_type == ScalarType::Long && static_shape(token) == Shape{1, int(proof->count)},
         "static B1 int64 tokens matching positions required");
    tokens_per_call_ = int(proof->count);
    need(tokens_per_call_ > 0 && tokens_per_call_ <= config_.capacity, "invalid static token count");
    const auto& output = metadata(program, Tid{program.output_map[0].idx});
    need(output.scalar_type == ScalarType::Float &&
         static_shape(output) == Shape{1, 1, config_.vocabulary}, "one FP32 vocabulary row required");

    std::unordered_map<std::string, Tid> names;
    for (const auto& named : program.named_slots) {
      if (named.slot.slot_type != SlotType::TensorSlot || named.slot.idx >= program.num_constant_tensors) continue;
      need(names.emplace(named.name, Tid{named.slot.idx}).second, "duplicate constant name");
    }
    const std::string suffix = "embed_tokens.weight";
    std::optional<std::string> prefix;
    for (const auto& named : names) {
      if (ends_with(named.first, suffix)) {
        need(!prefix, "ambiguous embedding namespace");
        prefix = named.first.substr(0, named.first.size() - suffix.size());
      }
    }
    need(prefix.has_value(), "missing canonical embedding namespace");
    std::set<uint32_t> used;
    auto bind = [&](const std::string& name) {
      std::string qualified = *prefix + name;
      if (olmoe() && name.rfind("lm_head.", 0) == 0) {
        need(ends_with(*prefix, "model."), "OLMoE embedding/head namespace differs");
        qualified = prefix->substr(0, prefix->size() - 6) + name;
      }
      const auto it = names.find(qualified);
      need(it != names.end() && used.insert(it->second.idx).second, "missing/aliased parameter role");
      const auto& value = constants.get(it->second);
      need(value.is_available() && value.flags().row_contiguous, "parameters must be completed row-contiguous arrays");
      weights_.emplace(it->second.idx, value);
      return it->second;
    };
    auto norm = [&](const std::string& name, int width = 0) {
      const Tid id = bind(name);
      need(w(id).dtype() == float16 && w(id).shape() == Shape{width ? width : config_.hidden_size}, "FP16 RMSNorm geometry differs");
      return id;
    };
    auto affine = [&](const std::string& name, int out, int in, int experts) {
      Affine ids{bind(name + ".weight"), bind(name + ".scales"), bind(name + ".biases"), in, out, experts, 0, 0};
      const auto& q = w(ids.q);
      const auto& scales = w(ids.scales);
      const auto& biases = w(ids.biases);
      need(q.dtype() == uint32 && scales.dtype() == float16 && biases.dtype() == float16,
           "original U32/FP16 affine Q/S/B required");
      std::optional<std::pair<int, int>> quantization;
      for (const auto& instruction : program.instruction_chains[program.main_chain_idx]) {
        if (experts && instruction.op == OpCode::GATHER_QMM) {
          const auto& n = std::get<GatherQmmNode>(instruction.node);
          if (n.w.idx != ids.q.idx) continue;
          need(!quantization && n.biases && n.biases->idx == ids.biases.idx && n.scales.idx == ids.scales.idx &&
               n.mode == "affine" && n.transpose, "expert affine graph operands differ");
          quantization = {{n.bits, n.group_size}};
        } else if (!experts && instruction.op == OpCode::QUANTIZED_MATMUL) {
          const auto& n = std::get<QuantizedMatmulNode>(instruction.node);
          if (n.w.idx != ids.q.idx) continue;
          need(!quantization && n.biases && n.biases->idx == ids.biases.idx && n.scales.idx == ids.scales.idx &&
               n.mode == "affine" && n.transpose, "linear affine graph operands differ");
          quantization = {{n.bits, n.group_size}};
        }
      }
      if (!quantization && olmoe() && name == "embed_tokens") {
        // The untied embedding has no QMM use. Bind its actual packed shape;
        // validate_graph below must prove selected-row DEQUANTIZE uses these
        // exact bits/group and original Q/S/B before accepting the program.
        need(q.ndim() == 2 && scales.ndim() == 2 && scales.shape(-1) > 0 &&
             int64_t(q.shape(-1)) * 32 % in == 0 && in % scales.shape(-1) == 0,
             "untied packed embedding dimensions differ");
        quantization = {{int(int64_t(q.shape(-1)) * 32 / in), in / scales.shape(-1)}};
      }
      need(quantization.has_value(), "missing packed projection/tied head graph use");
      ids.bits = quantization->first;
      ids.group_size = quantization->second;
      need((ids.bits == 4 || ids.bits == 8) && ids.group_size == 64 && in % 64 == 0,
           "only original affine4/64 or affine8/64 supported");
      Shape q_shape{out, in / (32 / ids.bits)}, sb_shape{out, in / 64};
      if (experts) { q_shape.insert(q_shape.begin(), experts); sb_shape.insert(sb_shape.begin(), experts); }
      need(q.shape() == q_shape && scales.shape() == sb_shape && biases.shape() == sb_shape,
           "packed parameter geometry differs");
      return ids;
    };
    embedding_ = affine("embed_tokens", config_.vocabulary, config_.hidden_size, 0);
    head_ = olmoe() ? affine("lm_head", config_.vocabulary, config_.hidden_size, 0) : embedding_;
    for (int i = 0; i < config_.layers; ++i) {
      const std::string base = "layers." + std::to_string(i) + ".";
      Layer layer;
      layer.input_norm = norm(base + "input_layernorm.weight");
      layer.post_norm = norm(base + "post_attention_layernorm.weight");
      layer.q = affine(base + "self_attn.q_proj", config_.query_heads * config_.head_dim, config_.hidden_size, 0);
      layer.k = affine(base + "self_attn.k_proj", config_.kv_heads * config_.head_dim, config_.hidden_size, 0);
      layer.v = affine(base + "self_attn.v_proj", config_.kv_heads * config_.head_dim, config_.hidden_size, 0);
      layer.o = affine(base + "self_attn.o_proj", config_.hidden_size, config_.hidden_size, 0);
      if (olmoe()) {
        layer.q_norm = norm(base + "self_attn.q_norm.weight", config_.query_heads * config_.head_dim);
        layer.k_norm = norm(base + "self_attn.k_norm.weight", config_.kv_heads * config_.head_dim);
      }
      layer.router = affine(base + (olmoe() ? "mlp.gate" : "block_sparse_moe.router.layer"), config_.num_experts, config_.hidden_size, 0);
      const auto mlp = base + (olmoe() ? "mlp.switch_mlp." : "block_sparse_moe.switch_mlp.");
      layer.gate = affine(mlp + "gate_proj", config_.intermediate_size, config_.hidden_size, config_.num_experts);
      layer.up = affine(mlp + "up_proj", config_.intermediate_size, config_.hidden_size, config_.num_experts);
      layer.down = affine(mlp + "down_proj", config_.hidden_size, config_.intermediate_size, config_.num_experts);
      layers_.push_back(layer);
    }
    final_norm_ = norm("norm.weight");
    for (const auto& named : names) {
      if (used.count(named.second.idx)) continue;
      const auto& value = constants.get(named.second);
      need((named.first.find("/_lifted_tensor_constant") != std::string::npos ||
            named.first.find("/_scalar_") != std::string::npos) &&
           value.ndim() == 0 && value.is_available(),
           "unexpected constant/unsupported model structure");
      Literal scalar;
      if (value.dtype() == float16) { scalar = {double(value.data<float16_t>()[0]), ScalarType::Half}; }
      else if (value.dtype() == float32) { scalar = {value.data<float>()[0], ScalarType::Float}; }
      else if (value.dtype() == int32) { scalar = {double(value.data<int32_t>()[0]), ScalarType::Int}; }
      else if (value.dtype() == int64) { scalar = {double(value.data<int64_t>()[0]), ScalarType::Long}; }
      else need(false, "unsupported literal precision");
      need(std::isfinite(scalar.value), "nonfinite graph literal");
      literals_.emplace(named.second.idx, scalar);
    }
    need(names.size() == program.num_constant_tensors, "unnamed constants");
    validate_graph(program);
  }

  static void validate_config(const NativeModelConfig& c) {
    need(is_native_moe_policy_architecture(c.architecture) &&
         c.quantization_layout == NativeQuantizationLayout::AffinePerProjection &&
         c.linear_quantization_bits == 0 && c.linear_quantization_group_size == 0 &&
         c.rotary_dimensions == 0 && c.layer_norm_eps == 0 &&
         c.tie_word_embeddings == (c.architecture == NativeModelArchitecture::GraniteMoE),
         "explicit supported MoE/affine-per-projection configuration required");
    for (int value : {c.layers, c.query_heads, c.kv_heads, c.head_dim, c.hidden_size,
                      c.intermediate_size, c.vocabulary, c.capacity, c.num_experts, c.experts_per_token})
      need(value > 0, "invalid model geometry");
    need(c.query_heads % c.kv_heads == 0 && c.head_dim % 2 == 0 &&
         int64_t(c.query_heads) * c.head_dim == c.hidden_size &&
         c.experts_per_token <= c.num_experts && c.hidden_size % 64 == 0 &&
         c.intermediate_size % 64 == 0 && int64_t(c.capacity) * c.experts_per_token <= INT32_MAX,
         "unsupported attention/expert geometry");
    for (float value : {c.rms_norm_eps, c.embedding_multiplier, c.residual_multiplier,
                        c.attention_multiplier, c.logits_scaling})
      need(std::isfinite(value) && value > 0, "invalid normalization/scaling configuration");
    need(std::isfinite(c.rope_theta) && c.rope_theta > 1, "invalid original RoPE theta");
    if (c.architecture == NativeModelArchitecture::OLMoE) {
      need(c.embedding_multiplier == 1 && c.residual_multiplier == 1 && c.logits_scaling == 1 &&
           c.attention_multiplier == float(1.0 / std::sqrt(double(c.head_dim))),
           "OLMoE requires original unscaled residual/embedding/head and head-dimension attention scale");
    }
  }

  bool matches(const NativeModelConfig& c) const {
    const auto& a = config_;
    return a.architecture == c.architecture && a.quantization_layout == c.quantization_layout &&
        a.layers == c.layers && a.query_heads == c.query_heads && a.kv_heads == c.kv_heads &&
        a.head_dim == c.head_dim && a.hidden_size == c.hidden_size && a.intermediate_size == c.intermediate_size &&
        a.vocabulary == c.vocabulary && a.capacity == c.capacity && a.rms_norm_eps == c.rms_norm_eps &&
        a.rope_theta == c.rope_theta && a.linear_quantization_bits == c.linear_quantization_bits &&
        a.linear_quantization_group_size == c.linear_quantization_group_size &&
        a.rotary_dimensions == c.rotary_dimensions && a.layer_norm_eps == c.layer_norm_eps &&
        a.num_experts == c.num_experts && a.experts_per_token == c.experts_per_token &&
        a.embedding_multiplier == c.embedding_multiplier && a.residual_multiplier == c.residual_multiplier &&
        a.attention_multiplier == c.attention_multiplier && a.logits_scaling == c.logits_scaling &&
        a.tie_word_embeddings == c.tie_word_embeddings;
  }

  template <class Silu>
  std::optional<Tensor> run(const Tensor& input_ids, const std::vector<int32_t>& positions,
      MLXSequenceCache& cache, bool cache_only, Silu&&,
      const ::mlx::core::Stream& stream, const ResidentFeatureConfig* feature_config = nullptr,
      std::optional<Tensor>* feature_summary = nullptr, int active_tokens = 0,
      const ResidentAttentionLabelConfig* attention_label_config = nullptr,
      std::optional<Tensor>* attention_label_summary = nullptr,
      const ResidentGreedyDeviceInput* device_input = nullptr,
      bool allow_partial_cache_only = false) const {
    using namespace ::mlx::core;
    need(!feature_config && !feature_summary && !attention_label_config && !attention_label_summary &&
         cache.has_complete_history_positions() && !cache.has_attention_selection(),
         "full-history native execution without advisers/labels required");
    need(stream.device.type == Device::gpu && input_ids.dtype() == int64 &&
         input_ids.shape() == Shape{1, tokens_per_call_} && positions.size() == size_t(tokens_per_call_) &&
         cache.capacity() == config_.capacity && input_ids.is_available() && input_ids.flags().row_contiguous,
         "native input/cache ABI differs");
    need(!device_input || (tokens_per_call_ == 1 && !cache_only && active_tokens == 0 &&
         device_input->value_.dtype() == int64 && device_input->value_.shape() == Shape{1, 1}),
         "native device token requires admitted full-history T1");
    need(!allow_partial_cache_only || (cache_only && active_tokens > 0 && !device_input),
         "partial cache-only requires an admitted native submission");
    need(active_tokens >= 0 && active_tokens <= tokens_per_call_ &&
         (!active_tokens || !cache_only || allow_partial_cache_only),
         "invalid final live-prefix count");
    const int T = active_tokens ? active_tokens : tokens_per_call_;
    need(!cache.has_cold_history() || (olmoe() && T == 1 && !cache_only),
         "cold KV currently requires OLMoE native T1");
    const int start = cache.length();
    need(start >= 0 && start <= config_.capacity && T <= config_.capacity - start, "absolute capacity exceeded");
    for (int i = 0; i < tokens_per_call_; ++i) {
      const auto token = input_ids.data<int64_t>()[i];
      need(i < T ? positions[size_t(i)] == start + i && token >= 0 && token < config_.vocabulary
                 : positions[size_t(i)] == 0 && token == 0, "invalid absolute position/token/padding");
    }
    std::optional<MoEWeightPrefetchWorker> weight_prefetch;
    std::optional<MoEWeightPrefetchCoroutineWorker> coroutine_prefetch;
    if (start == 0 && T > 1) {
      if (weight_prefetch_mode_ == NativeMoEWeightPrefetchMode::WillNeed) weight_prefetch.emplace(weight_prefetch_scope_);
      if (weight_prefetch_mode_ == NativeMoEWeightPrefetchMode::Coroutine) {
        coroutine_prefetch.emplace(std::max(0, config_.layers - 1 - int(cache_only)),
            weight_prefetch_layer_budget_, weight_prefetch_limits_, weight_prefetch_scope_);
      }
    }
    const auto ids = device_input ? device_input->value_ :
        T == tokens_per_call_ ? input_ids : slice(input_ids, {0, 0}, {1, T}, stream);
    const std::vector<int32_t> live_positions(positions.begin(), positions.begin() + T);
    auto h = dequantize(take(w(embedding_.q), ids, 0, stream),
        take(w(embedding_.scales), ids, 0, stream), take(w(embedding_.biases), ids, 0, stream),
        embedding_.group_size, embedding_.bits, "affine", std::nullopt, float16, stream);
    if (!olmoe()) h = scale(h, config_.embedding_multiplier, stream);
    for (int i = 0; i < config_.layers; ++i) {
      if (coroutine_prefetch) coroutine_prefetch->advance_layer(i);
      const int next_layer = i + 1;
      if ((weight_prefetch || coroutine_prefetch) && next_layer < config_.layers &&
          !(cache_only && next_layer + 1 == config_.layers) &&
          (!coroutine_prefetch || coroutine_prefetch->consider_layer(next_layer))) {
        // The arrays and their backing constants remain owned by this handle.
        // Hint only the next layer's experts, inside the timed prefill call.
        const auto& next = layers_[size_t(next_layer)];
        MoEWeightPrefetchJob job;
        if (weight_prefetch_scope_ == NativeMoEWeightPrefetchScope::GPUComplement && olmoe()) {
          if (const auto* hook = get_native_moe_expert_hook()) {
            // Only copy frozen plan IDs; future routing is not evaluated here.
            // Overflow of an excluded expert remains unchanged GPU work and
            // obtains its original packed weights through normal demand reads.
            job.layout = make_moe_expert_hint_layout(
                config_.num_experts, hook->planned_ane_experts_for_prefetch(next_layer));
          }
        }
        size_t range_index = 0;
        for (const auto* affine : {&next.gate, &next.up, &next.down}) {
          for (const auto id : {affine->q, affine->scales, affine->biases}) {
            const auto& value = w(id);
            if (job.layout) {
              need(value.is_available() && value.flags().row_contiguous && value.ndim() == 3 &&
                   value.shape(0) == config_.num_experts &&
                   value.nbytes() % size_t(config_.num_experts) == 0,
                   "expert-page hints require the original contiguous expert axis");
            }
            job[range_index++] = {value.data<uint8_t>(), value.nbytes()};
          }
        }
        if (weight_prefetch) weight_prefetch->try_submit(job);
        if (coroutine_prefetch) coroutine_prefetch->try_submit(next_layer, job);
      }
      const auto& layer = layers_[size_t(i)];
      const auto normalized = fast::rms_norm(h, w(layer.input_norm), config_.rms_norm_eps, stream);
      // Only the final query contributes to the returned prediction. Keep all
      // last-layer K/V rows, but avoid Q, attention output and MoE for rows that
      // no later layer consumes. T1 and intermediate cache-only calls keep
      // their original path; the final live-prefix count, not static padding,
      // selects the query's original absolute RoPE position.
      const bool last_query_only = !cache_only && i + 1 == config_.layers && T > 1;
      const int query_tokens = last_query_only ? 1 : T;
      const int query_offset = last_query_only ? T - 1 : 0;
      const auto query_input = last_query_only
          ? slice(normalized, {0, T - 1, 0}, {1, T, config_.hidden_size}, stream)
          : normalized;
      auto q_projected = linear(query_input, layer.q, stream);
      auto k_projected = linear(normalized, layer.k, stream);
      if (olmoe()) {
        // Official OLMoE normalizes the entire projected Q/K width before
        // splitting heads. Per-head RMSNorm would be a different model.
        q_projected = fast::rms_norm(q_projected, w(layer.q_norm), config_.rms_norm_eps, stream);
        k_projected = fast::rms_norm(k_projected, w(layer.k_norm), config_.rms_norm_eps, stream);
      }
      auto q = transpose(reshape(q_projected, {1, query_tokens, config_.query_heads, config_.head_dim}, stream), {0, 2, 1, 3}, stream);
      auto k = transpose(reshape(k_projected, {1, T, config_.kv_heads, config_.head_dim}, stream), {0, 2, 1, 3}, stream);
      auto v = transpose(reshape(linear(normalized, layer.v, stream), {1, T, config_.kv_heads, config_.head_dim}, stream), {0, 2, 1, 3}, stream);
      q = fast::rope(q, config_.head_dim, false, config_.rope_theta, 1.0f, start + query_offset, std::nullopt, stream);
      k = fast::rope(k, config_.head_dim, false, config_.rope_theta, 1.0f, start, std::nullopt, stream);
      const bool layer_is_cold = cache.has_cold_history(i);
      const auto attended = layer_is_cold
          ? cache.update_cold_and_fetch(i, live_positions, k, v, stream)
          : cache.has_cold_history()
              ? cache.update_dense_t1_and_fetch(i, live_positions, k, v, stream)
              : cache.update_and_fetch(i, live_positions, k, v, stream);
      const Shape kv{1, config_.kv_heads, start + T - cache.layer_cold_prefix_length(i), config_.head_dim};
      need(attended.K.shape() == kv && attended.V.shape() == kv &&
           attended.K.dtype() == float16 && attended.V.dtype() == float16 && !attended.mask &&
           attended.kind == (T == 1 ? AttendSpec::Mask::None : AttendSpec::Mask::Causal),
           "ordered full-history FP16 KV required");
      if (cache_only && i + 1 == config_.layers) return std::nullopt;
      if (last_query_only) h = slice(h, {0, T - 1, 0}, {1, T, config_.hidden_size}, stream);
      // A last-row query can attend every committed key, including this
      // chunk's entire prefix. Its single-row attention needs no causal mask.
      auto attention = [&]() {
        if (layer_is_cold) {
          const auto& cold = cache.cold_layer(i);
          return cold_attention_.apply(contiguous(q, false, stream), cold.key_codes, cold.key_scales, cold.key_biases,
              cold.value_codes, cold.value_scales, cold.value_biases, attended.K, attended.V, stream, config_.attention_multiplier);
        }
        return fast::scaled_dot_product_attention(q, attended.K, attended.V,
            config_.attention_multiplier, query_tokens == 1 ? "" : "causal", std::nullopt, std::nullopt, false, stream);
      }();
      attention = reshape(transpose(attention, {0, 2, 1, 3}, stream), {1, query_tokens, config_.hidden_size}, stream);
      const auto projected_attention = linear(attention, layer.o, stream);
      h = add(h, olmoe() ? projected_attention : scale(projected_attention, config_.residual_multiplier, stream), stream);
      const auto post = fast::rms_norm(h, w(layer.post_norm), config_.rms_norm_eps, stream);
      h = moe(post, h, layer, i, query_tokens, stream);
    }
    auto last = slice(h, {0, h.shape(1) - 1, 0}, {1, h.shape(1), config_.hidden_size}, stream);
    last = fast::rms_norm(last, w(final_norm_), config_.rms_norm_eps, stream);
    const auto logits = linear(last, head_, stream);
    return astype(olmoe() ? logits : divide(logits, array(config_.logits_scaling, float16), stream), float32, stream);
  }

 private:
  bool olmoe() const { return config_.architecture == NativeModelArchitecture::OLMoE; }
  struct Affine { Tid q{}, scales{}, biases{}; int input = 0, output = 0, experts = 0, bits = 0, group_size = 0; };
  struct Layer { Tid input_norm{}, post_norm{}, q_norm{}, k_norm{}; Affine q, k, v, o, router, gate, up, down; };
  mutable trifuse_cold_kv_experiment::ColdKVAttention cold_attention_;
  NativeModelConfig config_;
  NativeMoEOperatorMode operator_mode_;
  NativeMoEWeightPrefetchMode weight_prefetch_mode_ = NativeMoEWeightPrefetchMode::Off;
  int weight_prefetch_layer_budget_ = 0;
  MoEWeightPrefetchLimits weight_prefetch_limits_;
  NativeMoEWeightPrefetchScope weight_prefetch_scope_ = NativeMoEWeightPrefetchScope::Full;
  int tokens_per_call_ = 0;
  Affine embedding_, head_;
  Tid final_norm_{};
  std::vector<Layer> layers_;
  std::unordered_map<uint32_t, Tensor> weights_;
  struct Literal { double value = 0; ScalarType dtype = ScalarType::Float; };
  std::unordered_map<uint32_t, Literal> literals_;
  using Function = std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
  mutable std::map<::mlx::core::Stream, Function> swiglu_;
#if MLX_VERSION_NUMERIC >= 32003
  mutable std::map<::mlx::core::Stream, ::mlx::core::fast::CustomKernelFunction> weighted_residual_;
  mutable std::map<::mlx::core::Stream, ::mlx::core::fast::CustomKernelFunction> router_;
#endif

  static void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(std::string("native Granite MoE: ") + message);
  }
  static bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
  }
  static const TensorMeta& metadata(const MLXProgram& p, Tid id) {
    need(id.idx < p.tensor_meta.size() && p.tensor_meta[id.idx], "missing tensor metadata");
    return *p.tensor_meta[id.idx];
  }
  static ::mlx::core::Shape static_shape(const TensorMeta& meta) {
    ::mlx::core::Shape result;
    for (const auto& dim : meta.shape) { need(!dim.is_dynamic(), "dynamic shape unsupported"); result.push_back(dim.value); }
    return result;
  }
  const Tensor& w(Tid id) const { return weights_.at(id.idx); }
  static Tensor scale(const Tensor& x, float value, const ::mlx::core::Stream& stream) {
    return ::mlx::core::multiply(x, ::mlx::core::array(value, x.dtype()), stream);
  }
  Tensor linear(const Tensor& x, const Affine& a, const ::mlx::core::Stream& stream) const {
    return ::mlx::core::quantized_matmul(x, w(a.q), w(a.scales), w(a.biases), true, a.group_size, a.bits, "affine", stream);
  }
  Tensor expert(const Tensor& x, const Tensor& indices, const Affine& a, bool sorted,
                const ::mlx::core::Stream& stream) const {
    return ::mlx::core::gather_qmm(x, w(a.q), w(a.scales), w(a.biases), std::nullopt,
        indices, true, a.group_size, a.bits, "affine",
#if MLX_VERSION_NUMERIC >= 32003
        std::nullopt, // Optional global scale added in MLX 0.32.3; affine Q/S/B has none.
#endif
        sorted, stream);
  }
  static std::vector<Tensor> trace_swiglu(const std::vector<Tensor>& inputs) {
    using namespace ::mlx::core;
    return {multiply(multiply(inputs.at(0), sigmoid(inputs.at(0))), inputs.at(1))};
  }
  Tensor fused_swiglu(const Tensor& gate, const Tensor& up, const ::mlx::core::Stream& stream) const {
    using namespace ::mlx::core;
    // Match official mlx_lm.models.activations.swiglu, including fusion and
    // activation dtype. No all-expert floating-point weight materialization.
    const Stream previous = default_stream(stream.device);
    struct Restore { Stream previous; ~Restore() { set_default_stream(previous); } } restore{previous};
    StreamContext context(stream);
    auto it = swiglu_.find(stream);
    if (it == swiglu_.end()) it = swiglu_.emplace(stream, compile(&trace_swiglu, true)).first;
    return it->second({gate, up}).at(0);
  }
#if MLX_VERSION_NUMERIC >= 32003
  Tensor fused_weighted_residual(const Tensor& output, const Tensor& gates, const Tensor& residual,
                                int T, const ::mlx::core::Stream& stream) const {
    using namespace ::mlx::core;
    const int H = config_.hidden_size;
    const int K = config_.experts_per_token;
    need(K > 0 && K <= 8 && output.dtype() == float16 && gates.dtype() == float32 &&
         residual.dtype() == float16 && output.shape() == Shape{1, T, K, H} &&
         gates.shape() == Shape{1, T, K} && residual.shape() == Shape{1, T, H} &&
         residual.size() <= std::numeric_limits<int>::max(), "weighted residual geometry/dtype differs");
    auto it = weighted_residual_.find(stream);
    if (it == weighted_residual_.end()) {
      CompileOptions options;
      options.math_mode = MathMode::Safe;
      it = weighted_residual_.emplace(stream, fast::metal_kernel(
          "trifuse_granite_weighted_residual",
          {"expert_output", "gates", "residual", "residual_scale"}, {"result"},
          kGraniteWeightedResidualSource, "", true, false, options)).first;
    }
    return it->second(
        {output, gates, residual, array(config_.residual_multiplier, float16)},
        {residual.shape()}, {float16}, {int(residual.size()), 1, 1}, {128, 1, 1},
        {{"H", H}, {"K", K}}, std::nullopt, false, stream).at(0);
  }
#endif
  std::pair<Tensor, Tensor> route(const Tensor& logits, int T,
                                const ::mlx::core::Stream& stream) const {
    using namespace ::mlx::core;
    const int E = config_.num_experts;
    const int K = config_.experts_per_token;
    if (olmoe()) {
      if (operator_mode_ == NativeMoEOperatorMode::Router) return olmoe_route(logits, T, E, K, stream);
      const auto probabilities = softmax(logits, -1, true, stream);
      const auto order = argpartition(negative(probabilities, stream), K - 1, -1, stream);
      const auto indices = slice(order, {0, 0, 0}, {1, T, K}, stream);
      return {indices, take_along_axis(probabilities, indices, -1, stream)};
    }
#if MLX_VERSION_NUMERIC >= 32003
    // Target the per-token launch overhead; keep the established sorted
    // prefill path outside this small-rank implementation selection.
    if (T == 1 && E <= 64 && K <= 8) {
      need(E > 0 && K > 0 && K <= E && logits.dtype() == float16 &&
           logits.shape() == Shape{1, T, E}, "router geometry/dtype differs");
      auto it = router_.find(stream);
      if (it == router_.end()) {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        it = router_.emplace(stream, fast::metal_kernel(
            "trifuse_granite_router", {"router_logits"}, {"indices", "gates"},
            kGraniteRouterSource, "", true, false, options)).first;
      }
      auto outputs = it->second({logits}, {{1, T, K}, {1, T, K}}, {uint32, float32},
          {32, T, 1}, {32, 1, 1}, {{"E", E}, {"K", K}}, std::nullopt, false, stream);
      return {std::move(outputs.at(0)), std::move(outputs.at(1))};
    }
#endif
    const auto order = argpartition(logits, -K, -1, stream);
    const auto indices = slice(order, {0, 0, E - K}, {1, T, E}, stream);
    const auto gates = softmax(astype(take_along_axis(logits, indices, -1, stream), float32, stream), -1, false, stream);
    return {indices, gates};
  }
  Tensor moe(const Tensor& x, const Tensor& residual, const Layer& layer, int layer_index, int T,
             const ::mlx::core::Stream& stream) const {
    using namespace ::mlx::core;
    const int K = config_.experts_per_token;
    const auto logits = linear(x, layer.router, stream);
    const auto [indices, gates] = route(logits, T, stream);
    if (olmoe() && T > 1) {
      if (auto* hook = get_native_moe_expert_hook()) {
        const NativeMoEExpertHook::GPUCallable gpu = [&](
            const Tensor& token_indices, const Tensor& expert_ids,
            bool experts_sorted) -> std::pair<Tensor, Tensor> {
          need(token_indices.ndim() == 1 && expert_ids.ndim() == 1 &&
               token_indices.shape() == expert_ids.shape() &&
               (token_indices.dtype() == uint32 || token_indices.dtype() == int32) &&
               (expert_ids.dtype() == uint32 || expert_ids.dtype() == int32),
               "expert hook route arrays must be parallel integer vectors");
          const int N = token_indices.shape(0);
          if (N == 0) return {zeros({0, config_.hidden_size}, x.dtype(), stream),
                              zeros({0}, uint32, stream)};
          const auto order = experts_sorted ? arange(N, uint32, stream)
                                            : argsort(expert_ids, -1, stream);
          const auto inverse = experts_sorted ? order : argsort(order, -1, stream);
          const auto sorted_experts = experts_sorted ? expert_ids : take(expert_ids, order, 0, stream);
          const auto sorted_tokens = experts_sorted ? token_indices : take(token_indices, order, 0, stream);
          const auto selected_inputs = take(
              reshape(x, {T, 1, config_.hidden_size}, stream), sorted_tokens, 0, stream);
          const auto up = expert(selected_inputs, sorted_experts, layer.up, true, stream);
          const auto gate = expert(selected_inputs, sorted_experts, layer.gate, true, stream);
          const auto output = expert(fused_swiglu(gate, up, stream),
                                     sorted_experts, layer.down, true, stream);
          return {reshape(output, {N, config_.hidden_size}, stream), inverse};
        };
        if (auto output = hook->execute(layer_index, T, x, residual, indices, gates, stream, gpu)) {
          need(output->shape() == residual.shape() && output->dtype() == residual.dtype(),
               "expert hook output geometry/dtype differs");
          return *output;
        }
      }
    }
    const bool sorted = int64_t(T) * K >= 64;
    auto inputs = reshape(x, {1, T, 1, 1, config_.hidden_size}, stream);
    auto selected = indices;
    std::optional<Tensor> inverse;
    if (sorted) {
      const auto flat = reshape(indices, {T * K}, stream);
      const auto sorted_order = argsort(flat, -1, stream);
      // argsort returns a permutation. Invert it directly instead of sorting
      // the same T*K integer positions a second time. Every slot is written
      // exactly once; expert grouping and gather_qmm's sorted fast path stay
      // unchanged.
      inverse = operator_mode_ == NativeMoEOperatorMode::Baseline
          ? argsort(sorted_order, -1, stream)
          : put_along_axis(zeros({T * K}, sorted_order.dtype(), stream),
              sorted_order, arange(T * K, sorted_order.dtype(), stream), 0, stream);
      inputs = take(reshape(x, {T, 1, config_.hidden_size}, stream),
          floor_divide(sorted_order, array(K, sorted_order.dtype()), stream), 0, stream);
      selected = take(flat, sorted_order, 0, stream);
    }
    const auto up = expert(inputs, selected, layer.up, sorted, stream);
    const auto gate = expert(inputs, selected, layer.gate, sorted, stream);
    auto output = expert(fused_swiglu(gate, up, stream), selected, layer.down, sorted, stream);
    if (sorted) output = reshape(take(output, *inverse, 0, stream), {1, T, K, 1, config_.hidden_size}, stream);
    output = reshape(output, {1, T, K, config_.hidden_size}, stream);
    if (olmoe()) {
      // MLX OLMoE's multiply and sum each return the activation dtype.
      // Granite's FP32 gates/reduction fusion is not equivalent here.
      const auto weighted = multiply(output, expand_dims(gates, -1, stream), stream);
      return add(residual, sum(weighted, -2, false, stream), stream);
    }
#if MLX_VERSION_NUMERIC >= 32003
    if (K <= 8) return fused_weighted_residual(output, gates, residual, T, stream);
#endif
    const auto weighted = multiply(astype(output, float32, stream), expand_dims(gates, -1, stream), stream);
    const auto reduced = astype(sum(weighted, -2, false, stream), float16, stream);
    return add(residual, scale(reduced, config_.residual_multiplier, stream), stream);
  }

  void validate_graph(const MLXProgram& program) const;
};

inline void NativeGraniteMoEPrefill::validate_graph(const MLXProgram& p) const {
  // Follow values through recycled TIDs, preserving their logical shapes and
  // roles. Unrecognized operators or writes to inputs/constants fail admission.
  // This validates the specific supported graph, not arbitrary graph equality.
  using Shape = position_proof_detail::Shape;
  enum class Kind {
    Constant, Literal, Tokens, Positions, Offset, EmbQ, EmbS, EmbB, Embedding,
    Hidden, AttentionNorm, Q, K, QNorm, KNorm, V, RotQ, RotK, Attention, ProjectedAttention,
    ScaledAttention, AfterAttention, MoENorm, Router, Partition, Selected,
    SelectedLogits, Probabilities, NegativeProbabilities, Gates, SortOrder, InverseOrder, SortedSelected, TokenOrder,
    SortedInputs, ExpertGate, ExpertUp, Activation, ExpertProduct, ExpertOutput,
    WeightedExperts, ReducedExperts, ScaledMoE, FinalNorm, Logits, ScaledLogits
  };
  struct Value {
    Kind kind = Kind::Constant;
    int layer = -1;
    Shape shape;
    ScalarType dtype = ScalarType::Half;
    double literal = 0;
    bool sorted = false;
  };
  std::unordered_map<uint32_t, Value> values;
  std::unordered_map<uint32_t, Value> scalar_values;
  auto value = [&](Tid id) -> Value {
    const auto it = values.find(id.idx);
    need(it != values.end(), "unproved graph input");
    return it->second;
  };
  auto write = [&](Tid id, Value v) {
    need(id.idx >= p.num_constant_tensors && id.idx != p.input_map[0].idx &&
         id.idx != p.input_map[1].idx, "graph writes a constant/input tensor");
    values.insert_or_assign(id.idx, std::move(v));
  };
  auto is = [](const Value& v, Kind kind, int layer) { return v.kind == kind && v.layer == layer; };
  auto axis = [&](int a, size_t rank) {
    const auto normalized = position_proof_detail::axis(a, rank);
    need(normalized.has_value(), "invalid graph axis"); return *normalized;
  };
  auto integer = [&](const std::variant<int64_t, Vid>& v) {
    const auto* literal = std::get_if<int64_t>(&v);
    need(literal != nullptr, "dynamic graph shape/index unsupported"); return *literal;
  };
  auto broadcast = [&](const Shape& a, const Shape& b) {
    Shape out(std::max(a.size(), b.size()), 1);
    for (size_t i = 0; i < out.size(); ++i) {
      const int x = i < a.size() ? a[a.size() - i - 1] : 1;
      const int y = i < b.size() ? b[b.size() - i - 1] : 1;
      need(x == y || x == 1 || y == 1, "invalid graph broadcast");
      out[out.size() - i - 1] = std::max(x, y);
    }
    return out;
  };
  auto shape_vector = [](const ::mlx::core::Shape& shape) { return Shape(shape.begin(), shape.end()); };
  for (const auto& weight : weights_) values.emplace(weight.first,
      Value{Kind::Constant, -1, shape_vector(weight.second.shape()), weight.second.dtype() == ::mlx::core::uint32 ? ScalarType::UInt32 : ScalarType::Half});
  for (const auto& literal : literals_) values.emplace(literal.first,
      Value{Kind::Literal, -1, {}, literal.second.dtype, literal.second.value});
  values.emplace(p.input_map[0].idx, Value{Kind::Tokens, -1, {1, tokens_per_call_}, ScalarType::Long});
  values.emplace(p.input_map[1].idx, Value{Kind::Positions, -1,
      shape_vector(static_shape(metadata(p, Tid{p.input_map[1].idx}))), metadata(p, Tid{p.input_map[1].idx}).scalar_type});
  struct Projection { const Affine* affine; int layer; Kind input; Kind output; };
  std::unordered_map<uint32_t, Projection> projections;
  projections.emplace(head_.q.idx, Projection{&head_, config_.layers, Kind::FinalNorm, Kind::Logits});
  struct NormRole { int layer; Kind input; Kind output; };
  std::unordered_map<uint32_t, NormRole> norms;
  for (int i = 0; i < config_.layers; ++i) {
    const auto& l = layers_[size_t(i)];
    norms.emplace(l.input_norm.idx, NormRole{i, Kind::Hidden, Kind::AttentionNorm});
    norms.emplace(l.post_norm.idx, NormRole{i, Kind::AfterAttention, Kind::MoENorm});
    if (olmoe()) {
      norms.emplace(l.q_norm.idx, NormRole{i, Kind::Q, Kind::QNorm});
      norms.emplace(l.k_norm.idx, NormRole{i, Kind::K, Kind::KNorm});
    }
    for (auto item : {Projection{&l.q, i, Kind::AttentionNorm, Kind::Q},
                     Projection{&l.k, i, Kind::AttentionNorm, Kind::K},
                     Projection{&l.v, i, Kind::AttentionNorm, Kind::V},
                     Projection{&l.o, i, Kind::Attention, Kind::ProjectedAttention},
                     Projection{&l.router, i, Kind::MoENorm, Kind::Router},
                     Projection{&l.gate, i, Kind::MoENorm, Kind::ExpertGate},
                     Projection{&l.up, i, Kind::MoENorm, Kind::ExpertUp},
                     Projection{&l.down, i, Kind::ExpertProduct, Kind::ExpertOutput}})
      projections.emplace(item.affine->q.idx, item);
  }
  std::unordered_map<uint32_t, size_t> uses;
  size_t embedding_lookups = 0, rope_calls = 0, attention_calls = 0, routers = 0, sums = 0;
  const int T = tokens_per_call_, H = config_.hidden_size, E = config_.num_experts, K = config_.experts_per_token;
  const bool sorted = int64_t(T) * K >= 64;
  for (const auto& instruction : p.instruction_chains[p.main_chain_idx]) {
    switch (instruction.op) {
      case OpCode::ID_COPY: {
        const auto& n = std::get<IdCopyNode>(instruction.node); write(n.out, value(n.x)); break;
      }
      case OpCode::RESHAPE: {
        const auto& n = std::get<ReshapeNode>(instruction.node); auto v = value(n.x);
        const auto shape = position_proof_detail::reshape(v.shape, n.shape);
        need(shape.has_value(), "invalid/dynamic reshape"); v.shape = *shape; write(n.out, v); break;
      }
      case OpCode::EXPAND_DIMS: {
        const auto& n = std::get<ExpandDimsNode>(instruction.node); auto v = value(n.x);
        v.shape.insert(v.shape.begin() + axis(n.axis, v.shape.size() + 1), 1); write(n.out, v); break;
      }
      case OpCode::SQUEEZE: {
        const auto& n = std::get<SqueezeNode>(instruction.node); auto v = value(n.x);
        std::set<size_t> axes;
        for (int a : n.dims) { const auto ax = axis(a, v.shape.size()); need(v.shape[ax] == 1, "non-singleton squeeze"); axes.insert(ax); }
        Shape shape;
        for (size_t a = 0; a < v.shape.size(); ++a) if (n.dims.empty() ? v.shape[a] != 1 : !axes.count(a)) shape.push_back(v.shape[a]);
        v.shape = std::move(shape); write(n.out, v); break;
      }
      case OpCode::TRANSPOSE: {
        const auto& n = std::get<TransposeNode>(instruction.node); auto v = value(n.x);
        need((v.kind == Kind::Q || v.kind == Kind::K || v.kind == Kind::QNorm ||
              v.kind == Kind::KNorm || v.kind == Kind::V || v.kind == Kind::Attention) &&
             n.perm == std::vector<int32_t>{0, 2, 1, 3}, "unsupported attention transpose");
        need(n.perm.size() == v.shape.size(), "transpose rank differs"); Shape shape; std::set<size_t> axes;
        for (int a : n.perm) { const auto ax = axis(a, v.shape.size()); need(axes.insert(ax).second, "duplicate transpose axis"); shape.push_back(v.shape[ax]); }
        v.shape = std::move(shape); write(n.out, v); break;
      }
      case OpCode::CONTIGUOUS: {
        const auto& n = std::get<ContiguousNode>(instruction.node); write(n.out, value(n.x)); break;
      }
      case OpCode::ASTYPE: {
        const auto& n = std::get<AsTypeNode>(instruction.node); auto v = value(n.x);
        const auto type = static_cast<ScalarType>(n.scalar_type);
        need(type == ScalarType::Float || type == ScalarType::Half || type == ScalarType::Long ||
             type == ScalarType::Int || type == ScalarType::UInt32, "unsupported graph cast");
        if (v.kind == Kind::Literal && type == ScalarType::Half)
          v.literal = double(::mlx::core::float16_t(v.literal));
        else if (v.kind == Kind::Literal && type == ScalarType::Float)
          v.literal = double(float(v.literal));
        v.dtype = type; write(n.out, v); break;
      }
      case OpCode::SLICE: {
        const auto& n = std::get<SliceNode>(instruction.node); auto v = value(n.x);
        const auto ax = axis(int(integer(n.axis)), v.shape.size()); const int length = v.shape[ax];
        const int64_t raw_start = integer(n.start), raw_stop = integer(n.stop);
        const int begin = int(std::clamp<int64_t>(raw_start < 0 ? raw_start + length : raw_start, 0, length));
        const int end = int(std::clamp<int64_t>(raw_stop < 0 ? raw_stop + length : raw_stop, 0, length));
        need(n.step == 1 && begin < end, "non-unit/empty slice unsupported");
        if (v.kind == Kind::Partition) {
          need(ax + 1 == v.shape.size() &&
               (olmoe() ? begin == 0 && end == K : begin == E - K && end == E), "router partition slice differs");
          v.kind = Kind::Selected;
        } else if (begin == 0 && end == length) {
          // Exported Python indexing may retain full-axis identity slices.
        } else if (v.kind == Kind::Positions) {
          need(begin == 0 && end == 1 && ax + 1 == v.shape.size(), "RoPE offset must be first actual input position");
          v.kind = Kind::Offset;
        } else if (v.kind == Kind::Hidden || v.kind == Kind::FinalNorm) {
          need(v.layer == config_.layers && ax == 1 && begin == T - 1 && end == T, "only final hidden row slicing supported");
        } else need(begin == 0 && end == length, "unexpected value slice");
        v.shape[ax] = end - begin; write(n.out, v); break;
      }
      case OpCode::TAKE: {
        const auto& n = std::get<TakeNode>(instruction.node); auto v = value(n.x);
        const auto ax = axis(n.axis, v.shape.size());
        if (n.index.kind == 0) {
          need((n.index.literal == 0 || n.index.literal == -1) &&
               (v.shape[ax] == 1 || (v.kind == Kind::Positions && n.index.literal == 0)), "unsupported scalar take");
          if (v.kind == Kind::Positions) v.kind = Kind::Offset;
          v.shape.erase(v.shape.begin() + ax);
        } else {
          need(n.index.kind == 2 && ax == 0, "only explicit first-axis tensor gather supported");
          const auto index = value(n.index.tid);
          if (n.x.idx == embedding_.q.idx || n.x.idx == embedding_.scales.idx || n.x.idx == embedding_.biases.idx) {
            need(index.kind == Kind::Tokens, "embedding rows must follow input IDs");
            v.kind = n.x.idx == embedding_.q.idx ? Kind::EmbQ : n.x.idx == embedding_.scales.idx ? Kind::EmbS : Kind::EmbB;
          } else if (v.kind == Kind::MoENorm) {
            need(sorted && is(index, Kind::TokenOrder, v.layer), "expert input sort indices differ");
            v.kind = Kind::SortedInputs; v.sorted = true;
          } else if (v.kind == Kind::Selected) {
            need(sorted && is(index, Kind::SortOrder, v.layer), "expert index sort differs");
            v.kind = Kind::SortedSelected; v.sorted = true;
          } else if (v.kind == Kind::ExpertOutput) {
            need(sorted && v.sorted && is(index, Kind::InverseOrder, v.layer), "expert unsort differs");
            v.sorted = false;
          } else need(false, "unsupported tensor gather");
          Shape shape = index.shape; shape.insert(shape.end(), v.shape.begin() + 1, v.shape.end()); v.shape = std::move(shape);
        }
        write(n.out, v); break;
      }
      case OpCode::ITEM_INT: {
        const auto& n = std::get<ItemIntNode>(instruction.node); auto v = value(n.x);
        need(v.kind == Kind::Offset && (v.shape.empty() || position_proof_detail::elements(v.shape) == std::optional<size_t>{1}),
             "scalar offset must derive from first position"); scalar_values.insert_or_assign(n.out.idx, v); break;
      }
      case OpCode::DEQUANTIZE: {
        const auto& n = std::get<DequantizeNode>(instruction.node);
        const auto q = value(n.w), scales = value(n.scales);
        need(q.kind == Kind::EmbQ && scales.kind == Kind::EmbS && n.biases && value(*n.biases).kind == Kind::EmbB &&
             n.bits == embedding_.bits && n.group_size == embedding_.group_size && n.mode == "affine" && !n.global_scale &&
             (!n.dtype || *n.dtype == int(ScalarType::Half)), "only selected original embedding rows may dequantize");
        need(q.shape == Shape{1, T, H / (32 / embedding_.bits)} && scales.shape == Shape{1, T, H / 64}, "embedding row geometry differs");
        write(n.out, Value{olmoe() ? Kind::Hidden : Kind::Embedding, 0, {1, T, H}}); ++embedding_lookups; break;
      }
      case OpCode::RMS_NORM: {
        const auto& n = std::get<RMSNormNode>(instruction.node); auto v = value(n.x);
        need(n.weight && n.eps == config_.rms_norm_eps && v.dtype == ScalarType::Half, "RMSNorm precision/epsilon differs");
        if (n.weight->idx == final_norm_.idx) {
          need(is(v, Kind::Hidden, config_.layers), "final norm is disconnected"); v.kind = Kind::FinalNorm;
        } else {
          const auto it = norms.find(n.weight->idx); need(it != norms.end(), "unbound norm");
          const auto& role = it->second;
          need(is(v, role.input, role.layer) && !v.shape.empty() &&
               v.shape.back() == w(*n.weight).shape(0), "norm/residual order or whole-width geometry differs");
          if (role.input == Kind::Q || role.input == Kind::K)
            need(v.shape == Shape{1, T, role.input == Kind::Q ? config_.query_heads * config_.head_dim : config_.kv_heads * config_.head_dim},
                 "OLMoE Q/K norm must precede the head split");
          v.kind = role.output;
        }
        ++uses[n.weight->idx]; write(n.out, v); break;
      }
      case OpCode::QUANTIZED_MATMUL: {
        const auto& n = std::get<QuantizedMatmulNode>(instruction.node); auto v = value(n.x);
        const auto it = projections.find(n.w.idx); need(it != projections.end(), "unbound QMM");
        const auto& role = it->second; const auto& a = *role.affine;
        need(!a.experts && is(v, role.input, role.layer) && v.dtype == ScalarType::Half &&
             !v.shape.empty() && v.shape.back() == a.input && n.scales.idx == a.scales.idx && n.biases && n.biases->idx == a.biases.idx &&
             n.bits == a.bits && n.group_size == a.group_size && n.mode == "affine" && n.transpose, "QMM role/format/input differs");
        v.kind = role.output; v.shape.back() = a.output; ++uses[a.q.idx]; write(n.out, v); break;
      }
      case OpCode::ROPE: {
        const auto& n = std::get<RopeNode>(instruction.node); auto v = value(n.x);
        const auto offset = n.offset.is_vid ? scalar_values.find(n.offset.vid.idx) : scalar_values.end();
        const auto origin = n.offset.is_vid ? (need(offset != scalar_values.end(), "unproved RoPE scalar"), offset->second) : value(n.offset.tid);
        need(origin.kind == Kind::Offset && !n.freqs && !n.traditional && n.dims == config_.head_dim &&
             n.base == config_.rope_theta && n.scale == 1 && v.dtype == ScalarType::Half &&
             (v.kind == (olmoe() ? Kind::QNorm : Kind::Q) || v.kind == (olmoe() ? Kind::KNorm : Kind::K)) &&
             v.shape == Shape{1, (v.kind == Kind::Q || v.kind == Kind::QNorm) ? config_.query_heads : config_.kv_heads, T, config_.head_dim}, "RoPE geometry/offset/config differs");
        v.kind = (v.kind == Kind::Q || v.kind == Kind::QNorm) ? Kind::RotQ : Kind::RotK; write(n.out, v); ++rope_calls; break;
      }
      case OpCode::UPDATE_AND_ATTEND: {
        const auto& n = std::get<UpdateAndAttendNode>(instruction.node);
        need(n.layer_id && n.scale && *n.scale == config_.attention_multiplier && n.out_dtype &&
             *n.out_dtype == int(ScalarType::Half), "attention identity/scaling/output dtype differs");
        const int layer = *n.layer_id; const auto q = value(n.q), k = value(n.k), v = value(n.v), pos = value(n.position);
        need(is(q, Kind::RotQ, layer) && is(k, Kind::RotK, layer) && is(v, Kind::V, layer) &&
             k.shape == v.shape && q.dtype == ScalarType::Half && k.dtype == ScalarType::Half &&
             v.dtype == ScalarType::Half && pos.kind == Kind::Positions, "attention QKV/positions disconnected");
        write(n.out, Value{Kind::Attention, layer, q.shape}); ++attention_calls; break;
      }
      case OpCode::ARG_PARTITION: {
        const auto& n = std::get<ArgPartitionNode>(instruction.node); auto v = value(n.x);
        need(v.kind == (olmoe() ? Kind::NegativeProbabilities : Kind::Router) &&
             v.shape == Shape{1, T, E} && axis(n.axis, 3) == 2 &&
             (olmoe() ? integer(n.kth) == K - 1 : integer(n.kth) == -K || integer(n.kth) == E - K), "router top-k partition differs");
        v.kind = Kind::Partition; v.dtype = ScalarType::UInt32; write(n.out, v); ++routers; break;
      }
      case OpCode::TAKE_ALONG_AXIS: {
        const auto& n = std::get<TakeAlongAxisNode>(instruction.node); auto v = value(n.x); const auto index = value(n.indices);
        need(v.kind == (olmoe() ? Kind::Probabilities : Kind::Router) && is(index, Kind::Selected, v.layer) && index.shape == Shape{1, T, K} && axis(n.axis, 3) == 2,
             "router selected logits must follow its top-k IDs");
        v.kind = olmoe() ? Kind::Gates : Kind::SelectedLogits; v.shape = index.shape; write(n.out, v); break;
      }
      case OpCode::SOFTMAX: {
        const auto& n = std::get<SoftmaxNode>(instruction.node); auto v = value(n.x);
        if (olmoe()) {
          need(v.kind == Kind::Router && v.dtype == ScalarType::Half && v.shape == Shape{1, T, E} &&
               axis(n.axis, 3) == 2 && n.precise, "OLMoE requires precise full-expert softmax in Half");
          v.kind = Kind::Probabilities; write(n.out, v); break;
        }
        need(v.kind == Kind::SelectedLogits && v.dtype == ScalarType::Float && v.shape == Shape{1, T, K} && axis(n.axis, 3) == 2,
             "router must softmax selected logits in FP32"); v.kind = Kind::Gates; write(n.out, v); break;
      }
      case OpCode::NEG: {
        const auto& n = std::get<NegNode>(instruction.node); auto v = value(n.x);
        need(olmoe() && v.kind == Kind::Probabilities && v.dtype == ScalarType::Half &&
             v.shape == Shape{1, T, E}, "only OLMoE negative full probabilities may partition");
        v.kind = Kind::NegativeProbabilities; write(n.out, v); break;
      }
      case OpCode::ARGSORT: {
        const auto& n = std::get<ArgsortNode>(instruction.node); auto v = value(n.x);
        need(sorted && v.shape == Shape{T * K} && axis(n.axis, 1) == 0 &&
             (v.kind == Kind::Selected || v.kind == Kind::SortOrder), "unsupported expert sort");
        v.kind = v.kind == Kind::Selected ? Kind::SortOrder : Kind::InverseOrder;
        v.dtype = ScalarType::UInt32; write(n.out, v); break;
      }
      case OpCode::FLOOR_DIVIDE: {
        const auto& n = std::get<FloorDivideNode>(instruction.node); auto v = value(n.a); const auto divisor = value(n.b);
        need(v.kind == Kind::SortOrder && divisor.kind == Kind::Literal && divisor.literal == K, "expert token permutation differs");
        v.kind = Kind::TokenOrder; write(n.out, v); break;
      }
      case OpCode::GATHER_QMM: {
        const auto& n = std::get<GatherQmmNode>(instruction.node); auto v = value(n.x);
        const auto it = projections.find(n.w.idx); need(it != projections.end(), "unbound expert QMM");
        const auto& role = it->second; const auto& a = *role.affine;
        need(a.experts == E && n.rhs_indices && !n.lhs_indices && n.biases && n.biases->idx == a.biases.idx &&
             n.scales.idx == a.scales.idx && n.bits == a.bits && n.group_size == 64 && n.mode == "affine" && n.transpose,
             "expert original packed operands differ");
        const auto ids = value(*n.rhs_indices);
        const bool actual_sorted = n.sorted_indices_flag ? integer(*n.sorted_indices_flag) != 0 : n.sorted_indices;
        const Shape expected_input = role.input == Kind::MoENorm
            ? (sorted ? Shape{T * K, 1, H} : Shape{1, T, 1, 1, H})
            : (sorted ? Shape{T * K, 1, config_.intermediate_size} : Shape{1, T, K, 1, config_.intermediate_size});
        need((ids.dtype == ScalarType::UInt32 || ids.dtype == ScalarType::Long || ids.dtype == ScalarType::Int) &&
             actual_sorted == sorted && is(ids, sorted ? Kind::SortedSelected : Kind::Selected, role.layer) &&
             is(v, role.input == Kind::MoENorm && sorted ? Kind::SortedInputs : role.input, role.layer) &&
             v.dtype == ScalarType::Half && v.shape == expected_input,
             "expert selection/input/sort order differs");
        v.kind = role.output; v.sorted = sorted; v.shape = ids.shape; v.shape.push_back(1); v.shape.push_back(a.output);
        ++uses[a.q.idx]; write(n.out, v); break;
      }
      case OpCode::SILU: {
        const auto& n = std::get<SiluNode>(instruction.node); auto v = value(n.x);
        need(v.kind == Kind::ExpertGate && v.dtype == ScalarType::Half, "expert SiLU role/precision differs");
        v.kind = Kind::Activation; write(n.out, v); break;
      }
      case OpCode::MULTIPLY: {
        const auto& n = std::get<MultiplyNode>(instruction.node); auto a = value(n.a), b = value(n.b);
        if (a.kind == Kind::Literal) std::swap(a, b);
        const auto shape = broadcast(a.shape, b.shape);
        if (b.kind == Kind::Literal) {
          float expected = 0;
          if (a.kind == Kind::Embedding) { expected = config_.embedding_multiplier; a.kind = Kind::Hidden; }
          else if (a.kind == Kind::ProjectedAttention) { expected = config_.residual_multiplier; a.kind = Kind::ScaledAttention; }
          else if (a.kind == Kind::ReducedExperts) { expected = config_.residual_multiplier; a.kind = Kind::ScaledMoE; }
          else need(false, "unexpected scalar multiplication");
          const double encoded = b.dtype == ScalarType::Half ? double(::mlx::core::float16_t(expected)) : double(expected);
          need(b.literal == encoded && a.dtype == ScalarType::Half, "model multiplier/precision differs");
        } else if ((a.kind == Kind::Activation && b.kind == Kind::ExpertUp) ||
                   (b.kind == Kind::Activation && a.kind == Kind::ExpertUp)) {
          need(a.layer == b.layer && a.shape == b.shape && a.dtype == ScalarType::Half && b.dtype == ScalarType::Half,
               "SwiGLU operands differ"); a.kind = Kind::ExpertProduct;
        } else {
          if (a.kind == Kind::Gates) std::swap(a, b);
          need(a.kind == Kind::ExpertOutput && !a.sorted && is(b, Kind::Gates, a.layer) &&
               a.shape == Shape{1, T, K, H} && b.shape == Shape{1, T, K, 1} &&
               a.dtype == (olmoe() ? ScalarType::Half : ScalarType::Float) && b.dtype == a.dtype,
               "routed expert weighting precision differs");
          a.kind = Kind::WeightedExperts;
        }
        a.shape = shape; write(n.out, a); break;
      }
      case OpCode::SUM: {
        const auto& n = std::get<SumNode>(instruction.node); auto v = value(n.x);
        need(v.kind == Kind::WeightedExperts && v.dtype == (olmoe() ? ScalarType::Half : ScalarType::Float) && !n.keepdims &&
             n.axes.size() == 1 && axis(n.axes[0], v.shape.size()) == 2, "expert reduction precision differs");
        v.shape.erase(v.shape.begin() + 2); v.kind = Kind::ReducedExperts; write(n.out, v); ++sums; break;
      }
      case OpCode::ADD: {
        const auto& n = std::get<AddNode>(instruction.node); auto a = value(n.a), b = value(n.b);
        need(a.layer == b.layer && a.dtype == ScalarType::Half && b.dtype == ScalarType::Half && a.shape == b.shape,
             "residual geometry/precision differs");
        if (a.kind == Kind::ScaledAttention || a.kind == Kind::ScaledMoE ||
            (olmoe() && (a.kind == Kind::ProjectedAttention || a.kind == Kind::ReducedExperts))) std::swap(a, b);
        if (a.kind == Kind::Hidden && b.kind == (olmoe() ? Kind::ProjectedAttention : Kind::ScaledAttention)) a.kind = Kind::AfterAttention;
        else { need(a.kind == Kind::AfterAttention && b.kind == (olmoe() ? Kind::ReducedExperts : Kind::ScaledMoE), "residual order differs"); a.kind = Kind::Hidden; ++a.layer; }
        write(n.out, a); break;
      }
      case OpCode::DIVIDE: {
        const auto& n = std::get<DivideNode>(instruction.node); auto v = value(n.a); const auto divisor = value(n.b);
        need(is(v, Kind::Logits, config_.layers) && divisor.kind == Kind::Literal && divisor.literal == config_.logits_scaling &&
             v.dtype == ScalarType::Half, "logits scale/order differs"); v.kind = Kind::ScaledLogits; write(n.out, v); break;
      }
      default: need(false, "operation outside explicit Granite MoE graph");
    }
  }
  const auto result = value(Tid{p.output_map[0].idx});
  need(is(result, olmoe() ? Kind::Logits : Kind::ScaledLogits, config_.layers) && result.dtype == ScalarType::Float &&
       result.shape == Shape{1, 1, config_.vocabulary}, "final scaled output is disconnected");
  need(embedding_lookups == 1 && rope_calls == size_t(config_.layers) * 2 && attention_calls == size_t(config_.layers) &&
       routers == size_t(config_.layers) && sums == size_t(config_.layers), "missing/repeated layer operation");
  for (const auto& item : projections) need(uses[item.first] == 1, "missing/repeated projection");
  for (const auto& item : norms) need(uses[item.first] == 1, "missing/repeated norm");
  need(uses[final_norm_.idx] == 1, "missing/repeated final norm");
}
} // namespace executorch::backends::mlx
