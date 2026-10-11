// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
#pragma once

#include "MLXPositionProof.h"
#include "MLXNativeLinear.h"
#include "MLXSequenceCache.h"
#include "resident_prefill.h"
#include "resident_features.h"
#include "resident_attention_labels.h"

#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_map>

namespace executorch::backends::mlx {

// Optional offline teacher math, exposed for an independent synthetic GPU
// fixture. Source tensors are the actual native post-RoPE Q and ordered full K;
// this recomputes FP32 probabilities, not the fused SDPA kernel's internals.
inline Tensor native_attention_block_scores(
    const Tensor& query, const Tensor& full_keys, int query_position,
    int query_heads, int kv_heads, int head_dim, int block_tokens,
    const ::mlx::core::Stream& stream) {
  using namespace ::mlx::core;
  if (stream.device.type != Device::gpu || query_position < 0 ||
      query_position > std::numeric_limits<int>::max() - 64 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      query_heads % kv_heads || block_tokens != 64 ||
      query.dtype() != full_keys.dtype() ||
      (query.dtype() != float16 && query.dtype() != bfloat16 && query.dtype() != float32) ||
      query.shape() != Shape{1, query_heads, 1, head_dim} ||
      full_keys.shape() != Shape{1, kv_heads, query_position + 1, head_dim})
    throw std::invalid_argument("resident attention labels: expected post-RoPE T1 Q and exact causal full K geometry");
  const int end = query_position + 1;
  const int groups = query_heads / kv_heads;
  auto label_q = reshape(astype(query, float32, stream), {kv_heads, groups, head_dim}, stream);
  auto label_k = transpose(reshape(astype(full_keys, float32, stream),
                                   {kv_heads, end, head_dim}, stream), {0, 2, 1}, stream);
  const float scale = 1.0f / std::sqrt(float(head_dim));
  auto scores = multiply(matmul(label_q, label_k, stream), array(scale, float32), stream);
  auto importance = max(softmax(scores, -1, true, stream), 1, false, stream);
  const int blocks = 1 + (end - 1) / block_tokens;
  const int padding = (block_tokens - end % block_tokens) % block_tokens;
  if (padding) {
    importance = pad(importance, std::vector<std::pair<int, int>>{{0, 0}, {0, padding}},
                     array(0.0f, float32), "constant", stream);
  }
  return sum(reshape(importance, {kv_heads, blocks, block_tokens}, stream), 2, false, stream);
}

// Explicit, narrow standard-Llama adapter. Operations mirror the public native
// MLX Llama implementation (mlx-swift-lm/Libraries/MLXLLM/Models/Llama.swift).
// It reuses this handle's actual constant arrays and this session's cache. No
// weight reload, host KV export, or model-name-based hardware selection occurs.
// Eligibility checks deliberately reject unsupported structure; they are not a
// proof of arbitrary graph equivalence. Full-model numerical validation remains
// required, notably for native fused RoPE versus exported FP16 trigonometry.
// The compatibility name also covers an explicitly opted-in T1 decode handle.
class NativeLlamaPrefill {
 public:
  NativeLlamaPrefill(
      const MLXProgram& program, const ConstantData& constants,
      NativeLlamaPrefillConfig config, const ::mlx::core::Stream& stream)
      : config_(config) {
    validate_config(config_);
    need(program.init_chain_idx == -1 && program.mutable_buffer_map.empty() &&
         program.input_map.size() == 2 && program.output_map.size() == 1 &&
         program.input_map[0].slot_type == SlotType::TensorSlot &&
         program.output_map[0].slot_type == SlotType::TensorSlot,
         "unsupported program boundary/init/mutable state");
    const auto proof = prove_input_positions(program);
    need(bool(proof) && proof->update_count == size_t(config_.layers),
         "position chain does not prove native absolute positions");
    const auto token_tid = program.input_map[0].idx;
    need(token_tid < program.tensor_meta.size() && program.tensor_meta[token_tid].has_value(), "missing token metadata");
    const auto& token_meta = *program.tensor_meta[token_tid];
    need(token_meta.scalar_type == ScalarType::Long && token_meta.shape.size() == 2 &&
         !token_meta.shape[0].is_dynamic() && !token_meta.shape[1].is_dynamic() &&
         token_meta.shape[0].value == 1 && token_meta.shape[1].value > 0 &&
         size_t(token_meta.shape[1].value) == proof->count,
         "native Llama requires static B1 int64 tokens matching positions");
    tokens_per_call_ = token_meta.shape[1].value;
    need(tokens_per_call_ <= config_.capacity, "native Llama call exceeds original capacity");
    const auto output_tid = program.output_map[0].idx;
    need(output_tid < program.tensor_meta.size() && program.tensor_meta[output_tid].has_value(),
         "missing native Llama output metadata");
    const auto& output_meta = *program.tensor_meta[output_tid];
    need(output_meta.scalar_type == ScalarType::Float && output_meta.shape.size() == 3 &&
         !output_meta.shape[0].is_dynamic() && !output_meta.shape[1].is_dynamic() &&
         !output_meta.shape[2].is_dynamic() && output_meta.shape[0].value == 1 &&
         output_meta.shape[1].value == 1 && output_meta.shape[2].value == config_.vocabulary,
         "native Llama requires one FP32 full-vocabulary output row");

    std::unordered_map<std::string, Tid> names;
    for (const auto& named : program.named_slots) {
      if (named.slot.slot_type != SlotType::TensorSlot || named.slot.idx >= program.num_constant_tensors) continue;
      need(names.emplace(named.name, Tid{named.slot.idx}).second, "duplicate constant name");
    }
    const std::string embed_suffix = ".embed_tokens.weight";
    std::optional<std::string> prefix;
    for (const auto& item : names) {
      if (ends_with(item.first, embed_suffix)) {
        need(!prefix, "ambiguous embedding namespace");
        prefix = item.first.substr(0, item.first.size() - embed_suffix.size()) + ".";
      }
    }
    need(prefix.has_value(), "missing standard embedding namespace");
    std::set<uint32_t> used;
    std::set<uint32_t> norm_ids, linear_ids;
    auto weight = [&](const std::string& suffix, const ::mlx::core::Shape& shape) -> Tid {
      const auto it = names.find(*prefix + suffix);
      need(it != names.end(), "missing standard Llama weight");
      need(used.insert(it->second.idx).second, "aliased or duplicate weight roles");
      const auto& value = constants.get(it->second);
      need(value.shape() == shape, "weight shape differs from explicit geometry");
      need(value.dtype() == ::mlx::core::float16 || value.dtype() == ::mlx::core::float32,
           "native prefill supports only dense FP16/FP32 weights");
      if (!dtype_) dtype_ = value.dtype();
      need(value.dtype() == *dtype_, "mixed weight dtypes are unsupported");
      return it->second;
    };
    embedding_ = weight("embed_tokens.weight", {config_.vocabulary, config_.hidden_size});
    linear_ids.insert(embedding_.idx); // Tied vocabulary head, checked below.
    auto projection = [&](const std::string& suffix, const ::mlx::core::Shape& shape) -> Tid {
      if (!config_.linear_quantization_bits) return weight(suffix, shape);
      need(*dtype_ == ::mlx::core::float16, "packed native path requires FP16 activation/KV");
      const std::string name = *prefix + suffix.substr(0, suffix.size() - 7) +
          ".parametrizations.weight.original0_to_packed";
      const auto found = names.find(name);
      need(found != names.end(), "missing packed standard projection");
      std::optional<QuantizedMatmulNode> node;
      for (const auto& instruction : program.instruction_chains[program.main_chain_idx]) {
        if (instruction.op != OpCode::QUANTIZED_MATMUL) continue;
        const auto& q = std::get<QuantizedMatmulNode>(instruction.node);
        if (q.w.idx != found->second.idx) continue;
        need(!node, "packed projection has multiple graph uses");
        node = q;
      }
      need(node && node->biases && node->bits == config_.linear_quantization_bits &&
           node->group_size == config_.linear_quantization_group_size &&
           node->mode == "affine" && node->transpose, "packed graph parameters differ from configuration");
      for (Tid id : {node->w, node->scales, *node->biases})
        need(id.idx < program.num_constant_tensors && used.insert(id.idx).second,
             "affine parameters must be distinct static constants");
      const auto& scales = constants.get(node->scales);
      need(scales.dtype() == *dtype_, "affine scales must preserve activation dtype");
      linears_.emplace(node->w.idx, NativeLinearWeight::affine(constants.get(node->w), scales,
          constants.get(*node->biases), shape[0], shape[1], node->bits, node->group_size));
      quantized_nodes_.emplace(node->w.idx, *node);
      return node->w;
    };
    for (int layer = 0; layer < config_.layers; ++layer) {
      const std::string base = "layers." + std::to_string(layer) + ".";
      Layer ids;
      ids.input_norm = weight(base + "input_layernorm.weight", {config_.hidden_size});
      ids.post_norm = weight(base + "post_attention_layernorm.weight", {config_.hidden_size});
      norm_ids.insert(ids.input_norm.idx); norm_ids.insert(ids.post_norm.idx);
      const int q_width = config_.query_heads * config_.head_dim;
      const int kv_width = config_.kv_heads * config_.head_dim;
      ids.q = projection(base + "self_attn.q_proj.weight", {q_width, config_.hidden_size});
      ids.k = projection(base + "self_attn.k_proj.weight", {kv_width, config_.hidden_size});
      ids.v = projection(base + "self_attn.v_proj.weight", {kv_width, config_.hidden_size});
      ids.o = projection(base + "self_attn.o_proj.weight", {config_.hidden_size, q_width});
      ids.gate = projection(base + "mlp.gate_proj.weight", {config_.intermediate_size, config_.hidden_size});
      ids.up = projection(base + "mlp.up_proj.weight", {config_.intermediate_size, config_.hidden_size});
      ids.down = projection(base + "mlp.down_proj.weight", {config_.hidden_size, config_.intermediate_size});
      for (Tid id : {ids.q, ids.k, ids.v, ids.o, ids.gate, ids.up, ids.down}) linear_ids.insert(id.idx);
      layers_.push_back(ids);
    }
    final_norm_ = weight("norm.weight", {config_.hidden_size});
    norm_ids.insert(final_norm_.idx);
    const auto inv = names.find(*prefix + "rotary_emb.inv_freq");
    need(inv != names.end() && used.insert(inv->second.idx).second, "missing default RoPE frequencies");
    const auto& inv_freq = constants.get(inv->second);
    need(inv_freq.dtype() == ::mlx::core::float32 &&
         inv_freq.shape() == ::mlx::core::Shape{config_.head_dim / 2} &&
         inv_freq.is_available() && inv_freq.flags().row_contiguous,
         "RoPE frequencies must be available dense FP32");
    for (int i = 0; i < config_.head_dim / 2; ++i) {
      const double expected = std::pow(double(config_.rope_theta), -2.0 * i / config_.head_dim);
      const float observed = inv_freq.data<float>()[i];
      need(std::isfinite(observed) && std::abs(double(observed) - expected) <=
           expected * 0.000002, "RoPE frequency differs from explicit default theta");
    }
    // Only the original scalar graph literals may be left over. Additional
    // norms, unused affine parameters, an untied head, etc. are rejected.
    size_t literals = 0;
    for (const auto& named : names) {
      if (used.count(named.second.idx)) continue;
      const auto& value = constants.get(named.second);
      need(named.first.find("/_lifted_tensor_constant") != std::string::npos &&
           value.dtype() == ::mlx::core::float32 && value.ndim() == 0,
           "unexpected constant: unsupported Llama structure");
      ++literals;
    }
    need(literals <= 2 && names.size() == program.num_constant_tensors,
         "unexpected/unnamed/aliased constants");
    validate_graph(program, norm_ids, linear_ids);
    // Reuse arrays, including mmap-backed storage where already provided by
    // load_constants. These transpose views do not copy or reload weights.
    for (uint32_t id : used) weights_.emplace(id, constants.get(Tid{id}));
    for (uint32_t id : linear_ids) {
      if (!linears_.count(id)) linears_.emplace(id, NativeLinearWeight::dense(constants.get(Tid{id}), stream));
    }
    need(quantized_nodes_.size() == (config_.linear_quantization_bits ? size_t(config_.layers) * 7 : 0),
         "native quantization must cover all attention/MLP projections");
  }

  static void validate_config(const NativeLlamaPrefillConfig& c) {
    need(c.architecture == NativeModelArchitecture::StandardLlamaTied &&
         c.rotary_dimensions == 0 && c.layer_norm_eps == 0,
         "standard Llama adapter requires its own architecture configuration");
    for (int value : {c.layers, c.query_heads, c.kv_heads, c.head_dim, c.hidden_size,
                      c.intermediate_size, c.vocabulary, c.capacity})
      need(value > 0, "invalid native geometry");
    need(c.head_dim % 2 == 0 && c.query_heads % c.kv_heads == 0 &&
         int64_t(c.query_heads) * c.head_dim == c.hidden_size &&
         int64_t(c.kv_heads) * c.head_dim <= INT32_MAX,
         "unsupported attention geometry");
    need((c.linear_quantization_bits == 0 && c.linear_quantization_group_size == 0) ||
         (c.linear_quantization_bits == 8 && c.linear_quantization_group_size == 32),
         "unsupported native linear quantization configuration");
    need(std::isfinite(c.rms_norm_eps) && c.rms_norm_eps > 0 &&
         std::isfinite(c.rope_theta) && c.rope_theta > 1,
         "invalid RMSNorm/RoPE configuration");
  }
  bool matches(const NativeLlamaPrefillConfig& c) const {
    const auto& a = config_;
    return a.architecture == c.architecture && a.rotary_dimensions == c.rotary_dimensions &&
        a.layer_norm_eps == c.layer_norm_eps &&
        a.layers == c.layers && a.query_heads == c.query_heads && a.kv_heads == c.kv_heads &&
        a.head_dim == c.head_dim && a.hidden_size == c.hidden_size &&
        a.intermediate_size == c.intermediate_size && a.vocabulary == c.vocabulary &&
        a.capacity == c.capacity && a.rms_norm_eps == c.rms_norm_eps && a.rope_theta == c.rope_theta &&
        a.linear_quantization_bits == c.linear_quantization_bits &&
        a.linear_quantization_group_size == c.linear_quantization_group_size;
  }

  template <class Silu>
  std::optional<Tensor> run(
      const Tensor& input_ids, const std::vector<int32_t>& positions,
      MLXSequenceCache& cache, bool cache_only, Silu&& silu,
      const ::mlx::core::Stream& stream,
      const ResidentFeatureConfig* feature_config = nullptr,
      std::optional<Tensor>* feature_summary = nullptr,
      int active_tokens = 0,
      const ResidentAttentionLabelConfig* attention_label_config = nullptr,
      std::optional<Tensor>* attention_label_summary = nullptr) const {
    using namespace ::mlx::core;
    need(stream.device.type == Device::gpu && input_ids.dtype() == int64 &&
         input_ids.shape() == Shape{1, tokens_per_call_} &&
         positions.size() == size_t(tokens_per_call_) &&
         cache.capacity() == config_.capacity, "native prefill input/cache ABI differs");
    need(active_tokens >= 0 && active_tokens <= tokens_per_call_,
         "native partial prefill count exceeds static input");
    const int T = active_tokens > 0 ? active_tokens : tokens_per_call_;
    if (active_tokens > 0) {
      need(!cache_only && !feature_config && !feature_summary && cache.has_full_history_storage(),
           "native partial prefill requires final full-history call without features");
    }
    const int start = cache.length();
    need(bool(attention_label_config) == bool(attention_label_summary),
         "incomplete resident attention label collector");
    std::vector<std::optional<Tensor>> attention_label_rows;
    if (attention_label_config) {
      attention_label_config->validate_for(config_.layers, T);
      need(tokens_per_call_ == 1 && active_tokens == 0 && !cache_only &&
           !feature_config && !feature_summary && cache.has_full_history_storage() &&
           !cache.has_attention_selection() && cache.attention_selection_layer_calls() == 0,
           "attention labels require native static T1 with never-sparse full history");
      attention_label_rows.resize(attention_label_config->layers.size());
      attention_label_summary->reset();
    }
    need(bool(feature_config) == bool(feature_summary), "incomplete resident feature collector");
    std::vector<std::optional<Tensor>> feature_rows;
    if (feature_config) {
      feature_config->validate_for(config_.layers, T);
      feature_rows.resize(feature_config->layers.size());
      feature_summary->reset();
    }
    auto finish_features = [&] {
      if (!feature_config) return;
      std::vector<Tensor> rows;
      rows.reserve(feature_rows.size());
      for (const auto& row : feature_rows) {
        need(row.has_value(), "selected resident feature layer was not collected");
        rows.push_back(*row);
      }
      *feature_summary = contiguous(stack(rows, 0, stream), false, stream);
    };
    need(start >= 0 && start <= config_.capacity && T <= config_.capacity - start,
         "native prefill exceeds original absolute capacity");
    for (int i = 0; i < T; ++i)
      need(positions[size_t(i)] == start + i, "native prefill positions are not contiguous actual history");
    // Input is the already-available tensor bound from the retained Invocation.
    need(input_ids.is_available() && input_ids.flags().row_contiguous, "token input not host available");
    for (int i = 0; i < tokens_per_call_; ++i) {
      const int64_t token = input_ids.data<int64_t>()[i];
      if (i < T) {
        need(token >= 0 && token < config_.vocabulary, "native token outside vocabulary");
      } else {
        need(token == 0 && positions[size_t(i)] == 0, "native partial prefill padding must be zero");
      }
    }
    // Preserve validation of the complete static ET tensor, then construct the
    // actual native graph solely from its live prefix. Padding never reaches
    // embedding lookup, RoPE, attention or the K/V storage controller.
    const auto live_ids = active_tokens > 0 && T != tokens_per_call_
        ? slice(input_ids, {0, 0}, {1, T}, stream) : input_ids;
    std::vector<int32_t> live_positions;
    const std::vector<int32_t>* actual_positions = &positions;
    if (T != tokens_per_call_) {
      live_positions.assign(positions.begin(), positions.begin() + T);
      actual_positions = &live_positions;
    }
    auto h = take(w(embedding_), live_ids, 0, stream);
    const float scale = 1.0f / std::sqrt(float(config_.head_dim));
    for (int layer = 0; layer < config_.layers; ++layer) {
      const auto& ids = layers_[size_t(layer)];
      const auto norm = fast::rms_norm(h, w(ids.input_norm), config_.rms_norm_eps, stream);
      auto q = transpose(reshape(linear(norm, ids.q, stream), {1, T, config_.query_heads, config_.head_dim}, stream), {0, 2, 1, 3}, stream);
      auto k = transpose(reshape(linear(norm, ids.k, stream), {1, T, config_.kv_heads, config_.head_dim}, stream), {0, 2, 1, 3}, stream);
      auto v = transpose(reshape(linear(norm, ids.v, stream), {1, T, config_.kv_heads, config_.head_dim}, stream), {0, 2, 1, 3}, stream);
      if (feature_config) {
        const auto selected = std::find(feature_config->layers.begin(), feature_config->layers.end(), layer);
        if (selected != feature_config->layers.end()) {
          const int block = feature_config->block_tokens;
          const int blocks = T / block;
          // Reduce only already-needed normalized hidden/new K/new V. Do not
          // force Q in the cache-only last layer or read retained cache arrays.
          const auto hf = reshape(astype(norm, float32, stream), {blocks, block, config_.hidden_size}, stream);
          const auto kf = reshape(astype(k, float32, stream), {config_.kv_heads, blocks, block, config_.head_dim}, stream);
          const auto vf = reshape(astype(v, float32, stream), {config_.kv_heads, blocks, block, config_.head_dim}, stream);
          feature_rows[size_t(selected - feature_config->layers.begin())] = stack({
              mean(square(hf, stream), std::vector<int>{1, 2}, false, stream),
              max(abs(hf, stream), std::vector<int>{1, 2}, false, stream),
              mean(square(kf, stream), std::vector<int>{0, 2, 3}, false, stream),
              max(abs(kf, stream), std::vector<int>{0, 2, 3}, false, stream),
              mean(square(vf, stream), std::vector<int>{0, 2, 3}, false, stream),
              max(abs(vf, stream), std::vector<int>{0, 2, 3}, false, stream)}, 1, stream);
        }
      }
      q = fast::rope(q, config_.head_dim, false, config_.rope_theta, 1.0f, start, std::nullopt, stream);
      k = fast::rope(k, config_.head_dim, false, config_.rope_theta, 1.0f, start, std::nullopt, stream);
      auto attended = cache.update_and_fetch(layer, *actual_positions, k, v, stream);
      const auto expected_mask = T == 1 ? AttendSpec::Mask::None : AttendSpec::Mask::Causal;
      const Shape expected_kv{1, config_.kv_heads, cache.retained_length(layer), config_.head_dim};
      need(attended.kind == expected_mask && !attended.mask &&
           attended.K.dtype() == *dtype_ && attended.V.dtype() == *dtype_ &&
           attended.K.shape() == expected_kv && attended.V.shape() == expected_kv,
           "native Llama requires ordered same-dtype retained cache with the expected mask");
      if (attention_label_config) {
        const int end = start + 1;
        need(cache.retained_length(layer) == end &&
             attended.K.shape() == Shape{1, config_.kv_heads, end, config_.head_dim},
             "attention labels require every absolute history position exactly once");
        const auto selected = std::find(attention_label_config->layers.begin(), attention_label_config->layers.end(), layer);
        if (selected != attention_label_config->layers.end()) {
          attention_label_rows[size_t(selected - attention_label_config->layers.begin())] =
              native_attention_block_scores(q, attended.K, start, config_.query_heads,
                  config_.kv_heads, config_.head_dim, attention_label_config->block_tokens, stream);
        }
      }
      // Last-layer attention/MLP/head cannot affect any KV root and need not
      // be built for an intermediate cache-only chunk.
      if (cache_only && layer + 1 == config_.layers) {
        finish_features();
        return std::nullopt;
      }
      if (tokens_per_call_ == 1 && T == 1)
        attended = cache.select_for_attention(layer, attended, stream);
      auto attention = fast::scaled_dot_product_attention(q, attended.K, attended.V,
          scale, T == 1 ? "" : "causal", std::nullopt, std::nullopt, false, stream);
      attention = reshape(transpose(attention, {0, 2, 1, 3}, stream), {1, T, config_.hidden_size}, stream);
      h = add(h, linear(attention, ids.o, stream), stream);
      const auto post = fast::rms_norm(h, w(ids.post_norm), config_.rms_norm_eps, stream);
      h = add(h, linear(multiply(silu(linear(post, ids.gate, stream)),
                                     linear(post, ids.up, stream), stream), ids.down, stream), stream);
    }
    finish_features();
    if (attention_label_config) {
      std::vector<Tensor> rows;
      rows.reserve(attention_label_rows.size());
      for (const auto& row : attention_label_rows) {
        need(row.has_value(), "selected resident attention label layer was not collected");
        rows.push_back(*row);
      }
      *attention_label_summary = contiguous(stack(rows, 0, stream), false, stream);
    }
    auto last = slice(h, {0, T - 1, 0}, {1, T, config_.hidden_size}, stream);
    last = fast::rms_norm(last, w(final_norm_), config_.rms_norm_eps, stream);
    return astype(linear(last, embedding_, stream), float32, stream);
  }

 private:
  struct Layer { Tid input_norm, post_norm, q, k, v, o, gate, up, down; };
  NativeLlamaPrefillConfig config_;
  int tokens_per_call_ = 0;
  std::optional<::mlx::core::Dtype> dtype_;
  Tid embedding_, final_norm_;
  std::vector<Layer> layers_;
  std::unordered_map<uint32_t, Tensor> weights_;
  std::unordered_map<uint32_t, NativeLinearWeight> linears_;
  std::unordered_map<uint32_t, QuantizedMatmulNode> quantized_nodes_;
  static void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(std::string("native Llama prefill: ") + message);
  }
  static bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
  }
  const Tensor& w(Tid id) const { return weights_.at(id.idx); }
  Tensor linear(const Tensor& x, Tid weight, const ::mlx::core::Stream& stream) const {
    return linears_.at(weight.idx)(x, stream);
  }

  void validate_graph(const MLXProgram& p, const std::set<uint32_t>& norms,
                      const std::set<uint32_t>& linear_weights) const {
    std::unordered_map<uint32_t, uint32_t> transposes;
    std::unordered_map<uint32_t, size_t> norm_uses, linear_uses;
    std::set<int> layers;
    size_t silus = 0, other_matmuls = 0, cosines = 0, sines = 0, adds = 0, multiplies = 0;
    for (const auto& ins : p.instruction_chains[p.main_chain_idx]) {
      std::optional<std::pair<uint32_t, uint32_t>> transpose;
      switch (ins.op) {
        case OpCode::EXPAND_DIMS: case OpCode::TAKE: case OpCode::BROADCAST_TO:
        case OpCode::ASTYPE: case OpCode::RESHAPE: case OpCode::SLICE:
        case OpCode::NEG: case OpCode::CONCATENATE: break;
        case OpCode::TRANSPOSE: {
          const auto& n = std::get<TransposeNode>(ins.node);
          if (linear_weights.count(n.x.idx) && !quantized_nodes_.count(n.x.idx) && n.perm == std::vector<int32_t>{1, 0})
            transpose = {{n.out.idx, n.x.idx}};
          break;
        }
        case OpCode::RMS_NORM: {
          const auto& n = std::get<RMSNormNode>(ins.node);
          need(n.weight && norms.count(n.weight->idx) && n.eps == config_.rms_norm_eps,
               "unsupported normalization/epsilon");
          ++norm_uses[n.weight->idx]; break;
        }
        case OpCode::ADDMM: {
          const auto& n = std::get<AddmmNode>(ins.node);
          need(!n.bias, "linear bias is unsupported");
          const auto it = transposes.find(n.mat2.idx);
          if (it != transposes.end()) ++linear_uses[it->second];
          else ++other_matmuls;
          break;
        }
        case OpCode::QUANTIZED_MATMUL: {
          const auto& n = std::get<QuantizedMatmulNode>(ins.node);
          const auto it = quantized_nodes_.find(n.w.idx);
          need(it != quantized_nodes_.end(), "unbound quantized projection");
          const auto& expected = it->second;
          need(n.scales.idx == expected.scales.idx && n.biases && expected.biases &&
               n.biases->idx == expected.biases->idx && n.bits == expected.bits &&
               n.group_size == expected.group_size && n.mode == "affine" && n.transpose,
               "affine operands differ from bound projection");
          ++linear_uses[n.w.idx]; break;
        }
        case OpCode::UPDATE_AND_ATTEND: {
          const auto& n = std::get<UpdateAndAttendNode>(ins.node);
          need(n.layer_id && *n.layer_id >= 0 && *n.layer_id < config_.layers &&
               layers.insert(*n.layer_id).second && n.scale &&
               *n.scale == 1.0f / std::sqrt(float(config_.head_dim)), "attention layer/scale differs");
          break;
        }
        case OpCode::SILU: ++silus; break;
        case OpCode::COS: ++cosines; break;
        case OpCode::SIN: ++sines; break;
        case OpCode::ADD: ++adds; break;
        case OpCode::MULTIPLY: ++multiplies; break;
        default: need(false, "operation outside explicit standard-Llama adapter");
      }
      std::visit([&](const auto& n) {
        using N = std::decay_t<decltype(n)>;
        if constexpr (position_proof_detail::HasOut<N>::value) {
          if constexpr (std::is_same_v<std::decay_t<decltype(n.out)>, Tid>)
            transposes.erase(n.out.idx);
        }
        if constexpr (position_proof_detail::HasOuts<N>::value)
          for (const auto& id : n.outs) transposes.erase(id.idx);
        if constexpr (position_proof_detail::HasOutputs<N>::value)
          for (const auto& id : n.outputs) transposes.erase(id.idx);
      }, ins.node);
      if (transpose) transposes.insert_or_assign(transpose->first, transpose->second);
    }
    need(norm_uses.size() == norms.size() && linear_uses.size() == linear_weights.size(),
         "missing normalization/projection/tied head");
    for (uint32_t id : norms) need(norm_uses[id] == 1, "repeated/missing norm");
    for (uint32_t id : linear_weights) need(linear_uses[id] == 1, "repeated/missing projection/tied head");
    need(layers.size() == size_t(config_.layers) && silus == size_t(config_.layers) &&
         other_matmuls == 1 && cosines == 1 && sines == 1 &&
         adds == size_t(config_.layers) * 4 && multiplies == size_t(config_.layers) * 5 + 2,
         "unsupported attention/MLP/RoPE/residual structure");
  }
};
} // namespace executorch::backends::mlx
