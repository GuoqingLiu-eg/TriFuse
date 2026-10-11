// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
#pragma once

#include "MLXPositionProof.h"
#include "MLXSequenceCache.h"
#include "resident_prefill.h"
#include "resident_features.h"
#include "resident_attention_labels.h"

#include <cmath>
#include <limits>
#include <set>
#include <unordered_map>

namespace executorch::backends::mlx {

// An explicit dense GPT-NeoX parallel-residual adapter. It uses the loaded
// program's actual arrays and the same resident cache/lifecycle as Llama.
// Admission checks the lowered operator wiring, not a model-name allowlist.
// Fused native LayerNorm/RoPE/SDPA still require numerical qualification against
// an independent reference; structural admission is not bitwise equivalence.
class NativeNeoXPrefill {
 public:
  NativeNeoXPrefill(const MLXProgram& program, const ConstantData& constants,
                   NativeLlamaPrefillConfig config, const ::mlx::core::Stream& stream)
      : config_(config) {
    validate_config(config_);
    need(stream.device.type == ::mlx::core::Device::gpu,
         "native adapter requires the GPU stream");
    need(program.init_chain_idx == -1 && program.mutable_buffer_map.empty() &&
         program.input_map.size() == 2 && program.output_map.size() == 1 &&
         program.input_map[0].slot_type == SlotType::TensorSlot &&
         program.input_map[1].slot_type == SlotType::TensorSlot &&
         program.output_map[0].slot_type == SlotType::TensorSlot &&
         program.main_chain_idx >= 0 &&
         size_t(program.main_chain_idx) < program.instruction_chains.size(),
         "unsupported program boundary/init/mutable state");
    const auto proof = prove_input_positions(program);
    need(bool(proof) && proof->update_count == size_t(config_.layers),
         "position chain does not prove native absolute positions");
    const auto input = program.input_map[0].idx;
    need(input < program.tensor_meta.size() && program.tensor_meta[input],
         "missing token metadata");
    const auto& token = *program.tensor_meta[input];
    need(token.scalar_type == ScalarType::Long && token.shape.size() == 2 &&
         !token.shape[0].is_dynamic() && !token.shape[1].is_dynamic() &&
         token.shape[0].value == 1 && token.shape[1].value > 0 &&
         size_t(token.shape[1].value) == proof->count,
         "requires static B1 int64 tokens matching positions");
    tokens_per_call_ = token.shape[1].value;
    need(tokens_per_call_ <= config_.capacity, "call exceeds original capacity");
    const auto output = program.output_map[0].idx;
    need(output < program.tensor_meta.size() && program.tensor_meta[output],
         "missing output metadata");
    const auto& out = *program.tensor_meta[output];
    need(out.scalar_type == ScalarType::Float && out.shape.size() == 3 &&
         !out.shape[0].is_dynamic() && !out.shape[1].is_dynamic() &&
         !out.shape[2].is_dynamic() && out.shape[0].value == 1 &&
         out.shape[1].value == 1 && out.shape[2].value == config_.vocabulary,
         "requires one FP32 full-vocabulary output row");

    std::unordered_map<std::string, Tid> names;
    for (const auto& named : program.named_slots) {
      if (named.slot.slot_type != SlotType::TensorSlot ||
          named.slot.idx >= program.num_constant_tensors) continue;
      need(names.emplace(named.name, Tid{named.slot.idx}).second,
           "duplicate constant name");
    }
    const std::string suffix = "gpt_neox.embed_in.weight";
    std::optional<std::string> prefix;
    for (const auto& item : names) {
      if (ends_with(item.first, suffix)) {
        need(!prefix, "ambiguous embedding namespace");
        prefix = item.first.substr(0, item.first.size() - suffix.size());
      }
    }
    need(prefix.has_value(), "missing GPT-NeoX embedding namespace");
    std::set<uint32_t> used;
    auto weight = [&](const std::string& name, const ::mlx::core::Shape& shape) {
      const auto it = names.find(*prefix + name);
      need(it != names.end(), "missing GPT-NeoX parameter");
      need(used.insert(it->second.idx).second, "aliased or duplicate parameter roles");
      const auto& value = constants.get(it->second);
      need(value.shape() == shape && value.dtype() == ::mlx::core::float16,
           "parameter differs from dense FP16 geometry");
      weights_.emplace(it->second.idx, value);
      return it->second;
    };
    auto norm = [&](const std::string& name) {
      return AffineNorm{weight(name + ".weight", {config_.hidden_size}),
                        weight(name + ".bias", {config_.hidden_size})};
    };
    auto projection = [&](const std::string& name, int output_width, int input_width) {
      Linear result{weight(name + ".weight", {output_width, input_width}),
                    weight(name + ".bias", {output_width})};
      transposed_weights_.emplace(result.weight.idx,
          ::mlx::core::transpose(weights_.at(result.weight.idx), {1, 0}, stream));
      return result;
    };
    embedding_ = weight("gpt_neox.embed_in.weight", {config_.vocabulary, config_.hidden_size});
    for (int i = 0; i < config_.layers; ++i) {
      const auto base = "gpt_neox.layers." + std::to_string(i) + ".";
      Layer layer;
      layer.input_norm = norm(base + "input_layernorm");
      layer.post_norm = norm(base + "post_attention_layernorm");
      layer.qkv = projection(base + "attention.query_key_value", 3 * config_.hidden_size, config_.hidden_size);
      layer.o = projection(base + "attention.dense", config_.hidden_size, config_.hidden_size);
      layer.up = projection(base + "mlp.dense_h_to_4h", config_.intermediate_size, config_.hidden_size);
      layer.down = projection(base + "mlp.dense_4h_to_h", config_.hidden_size, config_.intermediate_size);
      layers_.push_back(layer);
    }
    final_norm_ = norm("gpt_neox.final_layer_norm");
    // Recent Transformers names the original checkpoint's embed_out as lm_head.
    // Bind the actual lowered name and keep this matrix untied from embed_in.
    head_ = weight("lm_head.weight", {config_.vocabulary, config_.hidden_size});
    transposed_weights_.emplace(head_.idx,
        ::mlx::core::transpose(weights_.at(head_.idx), {1, 0}, stream));
    const auto inv = names.find(*prefix + "gpt_neox.rotary_emb.inv_freq");
    need(inv != names.end() && used.insert(inv->second.idx).second,
         "missing shared partial-RoPE frequencies");
    inv_freq_ = inv->second;
    const auto& frequencies = constants.get(inv_freq_);
    weights_.emplace(inv_freq_.idx, frequencies);
    need(frequencies.dtype() == ::mlx::core::float32 &&
         frequencies.shape() == ::mlx::core::Shape{config_.rotary_dimensions / 2} &&
         frequencies.is_available() && frequencies.flags().row_contiguous,
         "partial-RoPE frequencies must be available dense FP32");
    for (int i = 0; i < config_.rotary_dimensions / 2; ++i) {
      const double expected = std::pow(double(config_.rope_theta),
                                      -2.0 * i / config_.rotary_dimensions);
      const double observed = frequencies.data<float>()[i];
      need(std::isfinite(observed) && std::abs(observed - expected) <= expected * 0.000002,
           "RoPE frequencies differ from explicit unscaled theta");
    }
    std::set<uint32_t> ones;
    for (const auto& item : names) {
      if (used.count(item.second.idx)) continue;
      const auto& value = constants.get(item.second);
      need(item.first.find("/_lifted_tensor_constant") != std::string::npos &&
           value.dtype() == ::mlx::core::float32 && value.ndim() == 0 &&
           value.is_available() && value.flags().row_contiguous &&
           value.data<float>()[0] == 1.0f && ones.insert(item.second.idx).second,
           "unexpected constant or non-unit RoPE scaling");
    }
    need(ones.size() <= 2 && used.size() + ones.size() == names.size() &&
         names.size() == program.num_constant_tensors,
         "unexpected/unnamed/aliased constants");
    validate_graph(program, ones);
  }

  static void validate_config(const NativeLlamaPrefillConfig& c) {
    need(c.architecture == NativeModelArchitecture::GPTNeoXParallel,
         "architecture is not explicit GPT-NeoX parallel residual");
    for (int v : {c.layers, c.query_heads, c.kv_heads, c.head_dim, c.hidden_size,
                  c.intermediate_size, c.vocabulary, c.capacity})
      need(v > 0, "invalid native geometry");
    need(c.query_heads == c.kv_heads && int64_t(c.query_heads) * c.head_dim == c.hidden_size &&
         c.hidden_size <= INT32_MAX / 3 && c.rotary_dimensions > 0 &&
         c.rotary_dimensions % 2 == 0 && c.rotary_dimensions <= c.head_dim,
         "unsupported MHA/partial-RoPE geometry");
    need(c.linear_quantization_bits == 0 && c.linear_quantization_group_size == 0 &&
         c.rms_norm_eps == 0 && std::isfinite(c.layer_norm_eps) && c.layer_norm_eps > 0 &&
         std::isfinite(c.rope_theta) && c.rope_theta > 1,
         "requires dense weights, affine LayerNorm and default RoPE");
  }

  bool matches(const NativeLlamaPrefillConfig& c) const {
    const auto& a = config_;
    return a.architecture == c.architecture && a.layers == c.layers &&
        a.query_heads == c.query_heads && a.kv_heads == c.kv_heads &&
        a.head_dim == c.head_dim && a.hidden_size == c.hidden_size &&
        a.intermediate_size == c.intermediate_size && a.vocabulary == c.vocabulary &&
        a.capacity == c.capacity && a.rms_norm_eps == c.rms_norm_eps &&
        a.layer_norm_eps == c.layer_norm_eps && a.rope_theta == c.rope_theta &&
        a.rotary_dimensions == c.rotary_dimensions &&
        a.linear_quantization_bits == c.linear_quantization_bits &&
        a.linear_quantization_group_size == c.linear_quantization_group_size;
  }

  template <class Silu>
  std::optional<Tensor> run(
      const Tensor& input_ids, const std::vector<int32_t>& positions,
      MLXSequenceCache& cache, bool cache_only, Silu&& /* unused_silu */,
      const ::mlx::core::Stream& stream,
      const ResidentFeatureConfig* feature_config = nullptr,
      std::optional<Tensor>* feature_summary = nullptr, int active_tokens = 0,
      const ResidentAttentionLabelConfig* attention_label_config = nullptr,
      std::optional<Tensor>* attention_label_summary = nullptr) const {
    using namespace ::mlx::core;
    need(!feature_config && !feature_summary && !attention_label_config && !attention_label_summary,
         "feature and teacher collectors are not part of this adapter");
    need(stream.device.type == Device::gpu && input_ids.dtype() == int64 &&
         input_ids.shape() == Shape{1, tokens_per_call_} &&
         positions.size() == size_t(tokens_per_call_) && cache.capacity() == config_.capacity &&
         cache.has_full_history_storage() && !cache.has_attention_selection() &&
         cache.attention_selection_layer_calls() == 0,
         "requires matching static ABI and never-sparse full-history cache");
    need(active_tokens >= 0 && active_tokens <= tokens_per_call_ &&
         (!active_tokens || !cache_only), "invalid final partial-prefill count");
    const int T = active_tokens ? active_tokens : tokens_per_call_;
    const int start = cache.length();
    need(start >= 0 && start <= config_.capacity && T <= config_.capacity - start,
         "native call exceeds original absolute capacity");
    need(input_ids.is_available() && input_ids.flags().row_contiguous,
         "token input must already be host available");
    for (int i = 0; i < tokens_per_call_; ++i) {
      const auto token = input_ids.data<int64_t>()[i];
      if (i < T) {
        need(positions[size_t(i)] == start + i && token >= 0 && token < config_.vocabulary,
             "invalid token or non-contiguous absolute position");
      } else {
        need(token == 0 && positions[size_t(i)] == 0, "partial-prefill padding must be zero");
      }
    }
    const auto ids = T == tokens_per_call_ ? input_ids : slice(input_ids, {0, 0}, {1, T}, stream);
    const std::vector<int32_t> live_positions(positions.begin(), positions.begin() + T);
    auto h = take(w(embedding_), ids, 0, stream);
    const float scale = 1.0f / std::sqrt(float(config_.head_dim));
    // Preserve HF's FP16 cosine/sine and product boundaries. Fused fast RoPE
    // retains more precision internally, which changes this model's FP16 KV.
    const auto frequency = multiply(expand_dims(arange(start, start + T, 1, float32, stream), -1, stream),
                                    expand_dims(w(inv_freq_), 0, stream), stream);
    const auto phase = concatenate({frequency, frequency}, -1, stream);
    const Shape rotation_shape{1, 1, T, config_.rotary_dimensions};
    const auto cosine = reshape(astype(cos(phase, stream), float16, stream), rotation_shape, stream);
    const auto sine = reshape(astype(sin(phase, stream), float16, stream), rotation_shape, stream);
    const auto rotate = [&](const Tensor& x) {
      const int R = config_.rotary_dimensions;
      const auto first = slice(x, {0,0,0,0}, {1,config_.query_heads,T,R / 2}, stream);
      const auto second = slice(x, {0,0,0,R / 2}, {1,config_.query_heads,T,R}, stream);
      const auto rotated = concatenate({negative(second, stream), first}, -1, stream);
      const auto rotary = slice(x, {0,0,0,0}, {1,config_.query_heads,T,R}, stream);
      const auto result = add(multiply(rotary, cosine, stream), multiply(rotated, sine, stream), stream);
      if (R == config_.head_dim) return result;
      return concatenate({result, slice(x, {0,0,0,R},
          {1,config_.query_heads,T,config_.head_dim}, stream)}, -1, stream);
    };
    for (int i = 0; i < config_.layers; ++i) {
      const auto& layer = layers_[size_t(i)];
      const auto norm = layer_norm(h, layer.input_norm, stream);
      // NeoX interleaves Q/K/V inside each head, not three hidden-size planes.
      auto qkv = transpose(reshape(linear(norm, layer.qkv, stream),
          {1, T, config_.query_heads, 3 * config_.head_dim}, stream), {0, 2, 1, 3}, stream);
      auto q = slice(qkv, {0, 0, 0, 0}, {1, config_.query_heads, T, config_.head_dim}, stream);
      auto k = slice(qkv, {0, 0, 0, config_.head_dim},
                     {1, config_.kv_heads, T, 2 * config_.head_dim}, stream);
      auto v = slice(qkv, {0, 0, 0, 2 * config_.head_dim},
                     {1, config_.kv_heads, T, 3 * config_.head_dim}, stream);
      q = rotate(q);
      k = rotate(k);
      const auto attended = cache.update_and_fetch(i, live_positions, k, v, stream);
      const Shape expected{1, config_.kv_heads, start + T, config_.head_dim};
      need(attended.kind == (T == 1 ? AttendSpec::Mask::None : AttendSpec::Mask::Causal) &&
           !attended.mask && attended.K.shape() == expected && attended.V.shape() == expected &&
           attended.K.dtype() == float16 && attended.V.dtype() == float16 &&
           cache.retained_length(i) == start + T,
           "cache must retain every absolute history position in order");
      if (cache_only && i + 1 == config_.layers) return std::nullopt;
      auto attention = fast::scaled_dot_product_attention(q, attended.K, attended.V, scale,
          T == 1 ? "" : "causal", std::nullopt, std::nullopt, false, stream);
      attention = reshape(transpose(attention, {0, 2, 1, 3}, stream),
                          {1, T, config_.hidden_size}, stream);
      attention = linear(attention, layer.o, stream);
      const auto mlp_input = layer_norm(h, layer.post_norm, stream);
      const auto mlp = linear(exact_gelu(linear(mlp_input, layer.up, stream), stream), layer.down, stream);
      // Preserve the exported (MLP + attention) + original hidden association.
      h = add(add(mlp, attention, stream), h, stream);
    }
    auto last = layer_norm(slice(h, {0, T - 1, 0}, {1, T, config_.hidden_size}, stream),
                           final_norm_, stream);
    return astype(matmul(last, transposed_weights_.at(head_.idx), stream), float32, stream);
  }

 private:
  struct AffineNorm { Tid weight, bias; };
  struct Linear { Tid weight, bias; };
  struct Layer { AffineNorm input_norm, post_norm; Linear qkv, o, up, down; };
  NativeLlamaPrefillConfig config_;
  int tokens_per_call_ = 0;
  Tid embedding_, head_, inv_freq_;
  AffineNorm final_norm_;
  std::vector<Layer> layers_;
  std::unordered_map<uint32_t, Tensor> weights_, transposed_weights_;
  static void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(std::string("native GPT-NeoX: ") + message);
  }
  static bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
  }
  const Tensor& w(Tid id) const { return weights_.at(id.idx); }
  Tensor linear(const Tensor& x, const Linear& projection, const ::mlx::core::Stream& stream) const {
    return ::mlx::core::addmm(w(projection.bias), x, transposed_weights_.at(projection.weight.idx),
                             1.0f, 1.0f, stream);
  }
  Tensor layer_norm(const Tensor& x, const AffineNorm& norm, const ::mlx::core::Stream& stream) const {
    using namespace ::mlx::core;
    // Half LayerNorm must round after the affine transform. MLX's half fast
    // kernel rounds normalization before gamma/beta; NeoX's learned affine
    // parameters can amplify that intermediate rounding. Float opmath keeps
    // the checkpoint's FP16 inputs/parameters, with one FP16 output rounding.
    return astype(fast::layer_norm(astype(x, float32, stream),
        astype(w(norm.weight), float32, stream), astype(w(norm.bias), float32, stream),
        config_.layer_norm_eps, stream), float16, stream);
  }
  static Tensor exact_gelu(const Tensor& x, const ::mlx::core::Stream& stream) {
    using namespace ::mlx::core;
    // PyTorch's Half GELU uses float opmath, then rounds the output to Half.
    // Do not substitute tanh GELU or round the erf argument to FP16 first.
    const auto f = astype(x, float32, stream);
    const auto cdf = add(array(1.0f, float32),
        erf(multiply(f, array(0.7071067811865475244f, float32), stream), stream), stream);
    return astype(multiply(multiply(f, cdf, stream), array(0.5f, float32), stream), x.dtype(), stream);
  }

  // Symbolic values are local to a single constructor and follow each actual
  // write to a recycled temporary TID. This proves the accepted wiring without
  // evaluating tensors, and prevents stale provenance from surviving slot reuse.
  enum class Kind {
    Tokens, Positions, PositionsBatched, PositionsColumn, PositionsFloat, CachePositions,
    Frequencies, FrequenciesBatched, FrequenciesColumn, AnglesColumn, Angles, DoubledAngles,
    One, CosFloat, SinFloat, CosHalf, SinHalf, CosHeads, SinHeads,
    Hidden, AttentionNorm, MLPNorm, FinalNorm, LastNorm, TransposedWeight,
    QKV, QKVShaped, QKVHeads, Q, K, V, QRot, KRot, QPass, KPass,
    QFirst, QSecond, KFirst, KSecond, NegQSecond, NegKSecond, RotateQ, RotateK,
    QCos, QSin, KCos, KSin, QRotated, KRotated, FullQ, FullK,
    AttentionHeads, AttentionShaped, AttentionFlat, AttentionBranch,
    MLPUp, GELU, MLPBranch, Branches, LogitsHalf, LogitsFloat
  };
  struct Value {
    Kind kind;
    int index = 0;
    bool operator==(const Value& b) const { return kind == b.kind && index == b.index; }
  };

  void validate_graph(const MLXProgram& p, const std::set<uint32_t>& ones) const {
    using K = Kind;
    std::unordered_map<uint32_t, Value> values;
    auto put = [&](Tid id, Value value) { values.insert_or_assign(id.idx, value); };
    auto get = [&](Tid id) -> Value {
      const auto it = values.find(id.idx);
      need(it != values.end(), "unproven value in native graph");
      return it->second;
    };
    auto is = [](Value v, K k, int index = 0) { return v.kind == k && v.index == index; };
    auto literal = [](const std::variant<int64_t, Vid>& value) {
      const auto* n = std::get_if<int64_t>(&value);
      need(n != nullptr, "dynamic shape/slice is unsupported");
      return *n;
    };
    auto shape_is = [](const std::vector<std::variant<int64_t, Vid>>& shape,
                       const std::vector<int32_t>& input, const std::vector<int32_t>& expected) {
      const auto resolved = position_proof_detail::reshape(input, shape);
      return resolved && *resolved == expected;
    };
    put(Tid{p.input_map[0].idx}, {K::Tokens});
    put(Tid{p.input_map[1].idx}, {K::Positions});
    put(inv_freq_, {K::Frequencies});
    for (auto id : ones) put(Tid{id}, {K::One});
    std::unordered_map<uint32_t, std::pair<AffineNorm, Value>> norms;
    std::unordered_map<uint32_t, std::pair<Linear, std::pair<Value, Value>>> projections;
    auto add_projection = [&](Linear linear, Value input, Value output) {
      projections.emplace(linear.weight.idx, std::make_pair(linear, std::make_pair(input, output)));
    };
    for (int i = 0; i < config_.layers; ++i) {
      const auto& l = layers_[size_t(i)];
      norms.emplace(l.input_norm.weight.idx, std::make_pair(l.input_norm, Value{K::AttentionNorm, i}));
      norms.emplace(l.post_norm.weight.idx, std::make_pair(l.post_norm, Value{K::MLPNorm, i}));
      add_projection(l.qkv, {K::AttentionNorm, i}, {K::QKV, i});
      add_projection(l.o, {K::AttentionFlat, i}, {K::AttentionBranch, i});
      add_projection(l.up, {K::MLPNorm, i}, {K::MLPUp, i});
      add_projection(l.down, {K::GELU, i}, {K::MLPBranch, i});
    }
    norms.emplace(final_norm_.weight.idx, std::make_pair(final_norm_, Value{K::FinalNorm, config_.layers}));
    std::set<uint32_t> used_norms, used_projections;
    std::set<int> used_attention, used_gelu, used_split;
    size_t cosine_count = 0, sine_count = 0, frequency_matmuls = 0, head_count = 0, embedding_count = 0;
    for (const auto& instruction : p.instruction_chains[p.main_chain_idx]) {
      // Native execution keeps constants and invocation inputs immutable.
      // A malformed interpreter graph must not redefine either storage role.
      auto writable = [&](Tid id) {
        need(id.idx >= p.num_constant_tensors && id.idx != p.input_map[0].idx &&
             id.idx != p.input_map[1].idx, "graph writes a constant or input tensor");
      };
      std::visit([&](const auto& node) {
        using N = std::decay_t<decltype(node)>;
        if constexpr (position_proof_detail::HasOut<N>::value) {
          if constexpr (std::is_same_v<std::decay_t<decltype(node.out)>, Tid>) writable(node.out);
        }
        if constexpr (position_proof_detail::HasOuts<N>::value)
          for (const auto& id : node.outs) writable(id);
        if constexpr (position_proof_detail::HasOutputs<N>::value)
          for (const auto& id : node.outputs) writable(id);
      }, instruction.node);
      switch (instruction.op) {
        case OpCode::ID_COPY: {
          const auto& n = std::get<IdCopyNode>(instruction.node);
          const auto value = get(n.x); put(n.out, value); break;
        }
        case OpCode::TAKE: {
          const auto& n = std::get<TakeNode>(instruction.node);
          if (n.x.idx == embedding_.idx) {
            need(n.axis == 0 && n.index.kind == 2 && is(get(n.index.tid), K::Tokens) &&
                 ++embedding_count == 1, "embedding lookup differs");
            put(n.out, {K::Hidden, 0});
          } else {
            need(is(get(n.x), K::PositionsBatched) && n.axis == 0 &&
                 n.index.kind == 0 && n.index.literal == 0, "unsupported take");
            put(n.out, {K::Positions});
          }
          break;
        }
        case OpCode::EXPAND_DIMS: {
          const auto& n = std::get<ExpandDimsNode>(instruction.node);
          const auto value = get(n.x);
          if (is(value, K::Positions) && n.axis == 0) put(n.out, {K::PositionsBatched});
          else if (is(value, K::PositionsBatched) && n.axis == 1) put(n.out, {K::PositionsColumn});
          else if (is(value, K::Frequencies) && n.axis == 0) put(n.out, {K::FrequenciesBatched});
          else if (is(value, K::FrequenciesBatched) && n.axis == 2) put(n.out, {K::FrequenciesColumn});
          else if (is(value, K::CosHalf) && n.axis == 1) put(n.out, {K::CosHeads});
          else if (is(value, K::SinHalf) && n.axis == 1) put(n.out, {K::SinHeads});
          else need(false, "unsupported broadcast axis or source");
          break;
        }
        case OpCode::BROADCAST_TO: {
          const auto& n = std::get<BroadcastToNode>(instruction.node);
          need(is(get(n.x), K::FrequenciesColumn) && n.shape.size() == 3 &&
               literal(n.shape[0]) == 1 &&
               (literal(n.shape[1]) == -1 || literal(n.shape[1]) == config_.rotary_dimensions / 2) &&
               literal(n.shape[2]) == 1, "unsupported frequency broadcast");
          put(n.out, {K::FrequenciesColumn}); break;
        }
        case OpCode::ASTYPE: {
          const auto& n = std::get<AsTypeNode>(instruction.node);
          const auto value = get(n.x);
          if (is(value, K::PositionsColumn) && n.scalar_type == int8_t(ScalarType::Float))
            put(n.out, {K::PositionsFloat});
          else if (is(value, K::CosFloat) && n.scalar_type == int8_t(ScalarType::Half))
            put(n.out, {K::CosHalf});
          else if (is(value, K::SinFloat) && n.scalar_type == int8_t(ScalarType::Half))
            put(n.out, {K::SinHalf});
          else if (is(value, K::LogitsHalf) && n.scalar_type == int8_t(ScalarType::Float))
            put(n.out, {K::LogitsFloat});
          else need(false, "unsupported dtype conversion");
          break;
        }
        case OpCode::TRANSPOSE: {
          const auto& n = std::get<TransposeNode>(instruction.node);
          if (transposed_weights_.count(n.x.idx) && n.perm == std::vector<int32_t>{1, 0}) {
            need(n.x.idx <= uint32_t(INT32_MAX), "weight id overflow");
            put(n.out, {K::TransposedWeight, int(n.x.idx)}); break;
          }
          const auto value = get(n.x);
          if (value.kind == K::QKVShaped && n.perm == std::vector<int32_t>{0, 2, 1, 3})
            put(n.out, {K::QKVHeads, value.index});
          else if (value.kind == K::AttentionHeads && n.perm == std::vector<int32_t>{0, 2, 1, 3})
            put(n.out, {K::AttentionShaped, value.index});
          else if (is(value, K::AnglesColumn) && n.perm == std::vector<int32_t>{0, 2, 1})
            put(n.out, {K::Angles});
          else need(false, "unsupported transpose geometry");
          break;
        }
        case OpCode::RESHAPE: {
          const auto& n = std::get<ReshapeNode>(instruction.node);
          const auto value = get(n.x);
          if (value.kind == K::QKV && shape_is(n.shape, {1, tokens_per_call_, 3 * config_.hidden_size},
              {1, tokens_per_call_, config_.query_heads, 3 * config_.head_dim}))
            put(n.out, {K::QKVShaped, value.index});
          else if (value.kind == K::AttentionShaped && shape_is(n.shape,
              {1, tokens_per_call_, config_.query_heads, config_.head_dim}, {1, tokens_per_call_, config_.hidden_size}))
            put(n.out, {K::AttentionFlat, value.index});
          else if (is(value, K::Positions) && shape_is(n.shape, {tokens_per_call_}, {tokens_per_call_, 1}))
            put(n.out, {K::CachePositions});
          else need(false, "unsupported reshape geometry");
          break;
        }
        case OpCode::SPLIT: {
          const auto& n = std::get<SplitNode>(instruction.node);
          const auto value = get(n.x);
          need(value.kind == K::QKVHeads && (n.axis == 3 || n.axis == -1) && n.outs.size() == 3 &&
               (n.sizes.size() == 1 || n.sizes.size() == 3) && used_split.insert(value.index).second,
               "QKV must split each head in Q/K/V order");
          for (const auto& size : n.sizes) need(literal(size) == config_.head_dim, "invalid QKV split size");
          need(n.outs[0].idx != n.outs[1].idx && n.outs[0].idx != n.outs[2].idx &&
               n.outs[1].idx != n.outs[2].idx, "aliased QKV outputs");
          put(n.outs[0], {K::Q, value.index}); put(n.outs[1], {K::K, value.index});
          put(n.outs[2], {K::V, value.index}); break;
        }
        case OpCode::SLICE: {
          const auto& n = std::get<SliceNode>(instruction.node);
          const auto value = get(n.x);
          const auto axis = literal(n.axis), start = literal(n.start), stop = literal(n.stop);
          need(n.step == 1, "unsupported slice step");
          const bool tail = stop == std::numeric_limits<int64_t>::max();
          if (value.kind == K::FinalNorm && axis == 1 &&
              (start == -1 || start == tokens_per_call_ - 1) && (tail || stop == tokens_per_call_)) {
            put(n.out, {K::LastNorm, value.index}); break;
          }
          need(axis == 3 || axis == -1, "partial-RoPE slice must use head dimension");
          const int r = config_.rotary_dimensions;
          if ((value.kind == K::Q || value.kind == K::K) && start == 0 && stop == r)
            put(n.out, {value.kind == K::Q ? K::QRot : K::KRot, value.index});
          else if ((value.kind == K::Q || value.kind == K::K) && start == r && (tail || stop == config_.head_dim))
            put(n.out, {value.kind == K::Q ? K::QPass : K::KPass, value.index});
          else if ((value.kind == K::QRot || value.kind == K::KRot) && start == 0 && stop == r / 2)
            put(n.out, {value.kind == K::QRot ? K::QFirst : K::KFirst, value.index});
          else if ((value.kind == K::QRot || value.kind == K::KRot) && start == r / 2 && (tail || stop == r))
            put(n.out, {value.kind == K::QRot ? K::QSecond : K::KSecond, value.index});
          else need(false, "unsupported partial-RoPE slice");
          break;
        }
        case OpCode::NEG: {
          const auto& n = std::get<NegNode>(instruction.node);
          const auto value = get(n.x);
          need(value.kind == K::QSecond || value.kind == K::KSecond, "unsupported negation");
          put(n.out, {value.kind == K::QSecond ? K::NegQSecond : K::NegKSecond, value.index}); break;
        }
        case OpCode::CONCATENATE: {
          const auto& n = std::get<ConcatenateNode>(instruction.node);
          need(n.tensors.size() == 2 && n.axis == -1, "unsupported concatenation");
          const auto a = get(n.tensors[0]), b = get(n.tensors[1]);
          need(a.index == b.index, "concatenation crosses model layers");
          if (is(a, K::Angles) && is(b, K::Angles)) put(n.out, {K::DoubledAngles});
          else if (a.kind == K::NegQSecond && b.kind == K::QFirst) put(n.out, {K::RotateQ, a.index});
          else if (a.kind == K::NegKSecond && b.kind == K::KFirst) put(n.out, {K::RotateK, a.index});
          else if (a.kind == K::QRotated && b.kind == K::QPass) put(n.out, {K::FullQ, a.index});
          else if (a.kind == K::KRotated && b.kind == K::KPass) put(n.out, {K::FullK, a.index});
          else need(false, "unsupported rotate-half or partial-RoPE concatenation");
          break;
        }
        case OpCode::COS: {
          const auto& n = std::get<CosNode>(instruction.node);
          need(is(get(n.x), K::DoubledAngles) && ++cosine_count == 1, "unsupported cosine source");
          put(n.out, {K::CosFloat}); break;
        }
        case OpCode::SIN: {
          const auto& n = std::get<SinNode>(instruction.node);
          need(is(get(n.x), K::DoubledAngles) && ++sine_count == 1, "unsupported sine source");
          put(n.out, {K::SinFloat}); break;
        }
        case OpCode::MULTIPLY: {
          const auto& n = std::get<MultiplyNode>(instruction.node);
          const auto a = get(n.a), b = get(n.b);
          if ((is(a, K::CosFloat) || is(a, K::SinFloat)) && is(b, K::One)) put(n.out, a);
          else if (a.kind == K::QRot && is(b, K::CosHeads)) put(n.out, {K::QCos, a.index});
          else if (a.kind == K::KRot && is(b, K::CosHeads)) put(n.out, {K::KCos, a.index});
          else if (a.kind == K::RotateQ && is(b, K::SinHeads)) put(n.out, {K::QSin, a.index});
          else if (a.kind == K::RotateK && is(b, K::SinHeads)) put(n.out, {K::KSin, a.index});
          else need(false, "unsupported RoPE multiplication");
          break;
        }
        case OpCode::ADD: {
          const auto& n = std::get<AddNode>(instruction.node);
          const auto a = get(n.a), b = get(n.b);
          need(a.index == b.index, "residual/RoPE addition crosses layers");
          if (a.kind == K::QCos && b.kind == K::QSin) put(n.out, {K::QRotated, a.index});
          else if (a.kind == K::KCos && b.kind == K::KSin) put(n.out, {K::KRotated, a.index});
          else if (a.kind == K::MLPBranch && b.kind == K::AttentionBranch) put(n.out, {K::Branches, a.index});
          else if (a.kind == K::Branches && b.kind == K::Hidden) put(n.out, {K::Hidden, a.index + 1});
          else need(false, "unsupported parallel-residual or RoPE addition");
          break;
        }
        case OpCode::LAYER_NORM: {
          const auto& n = std::get<LayerNormNode>(instruction.node);
          need(n.weight && n.bias && n.eps == config_.layer_norm_eps, "unsupported affine LayerNorm/epsilon");
          const auto it = norms.find(n.weight->idx);
          need(it != norms.end() && n.bias->idx == it->second.first.bias.idx &&
               used_norms.insert(n.weight->idx).second &&
               is(get(n.x), K::Hidden, it->second.second.index),
               "LayerNorm bias/input differs from parallel residual");
          put(n.out, it->second.second); break;
        }
        case OpCode::ADDMM: {
          const auto& n = std::get<AddmmNode>(instruction.node);
          const auto a = get(n.mat1), b = get(n.mat2);
          need(n.alpha == 1.0f && n.beta == 1.0f, "unsupported projection multiplier");
          if (is(a, K::FrequenciesColumn) && is(b, K::PositionsFloat)) {
            need(!n.bias && ++frequency_matmuls == 1, "unsupported RoPE frequency product");
            put(n.out, {K::AnglesColumn}); break;
          }
          need(b.kind == K::TransposedWeight, "unbound native projection");
          if (uint32_t(b.index) == head_.idx) {
            need(!n.bias && ++head_count == 1 &&
                 (is(a, K::LastNorm, config_.layers) ||
                  (tokens_per_call_ == 1 && is(a, K::FinalNorm, config_.layers))),
                 "untied vocabulary head/input differs");
            put(n.out, {K::LogitsHalf}); break;
          }
          const auto it = projections.find(uint32_t(b.index));
          need(it != projections.end() && n.bias && n.bias->idx == it->second.first.bias.idx &&
               a == it->second.second.first && used_projections.insert(uint32_t(b.index)).second,
               "projection bias/input/role differs");
          put(n.out, it->second.second.second); break;
        }
        case OpCode::GELU: {
          const auto& n = std::get<GeluNode>(instruction.node);
          const auto value = get(n.x);
          need(value.kind == K::MLPUp && n.approximate == "none" &&
               used_gelu.insert(value.index).second, "requires exact erf GELU once per layer");
          put(n.out, {K::GELU, value.index}); break;
        }
        case OpCode::UPDATE_AND_ATTEND: {
          const auto& n = std::get<UpdateAndAttendNode>(instruction.node);
          need(n.layer_id && *n.layer_id >= 0 && *n.layer_id < config_.layers && n.scale &&
               *n.scale == 1.0f / std::sqrt(float(config_.head_dim)) && n.out_dtype &&
               *n.out_dtype == int8_t(ScalarType::Half) && used_attention.insert(*n.layer_id).second &&
               is(get(n.q), K::FullQ, *n.layer_id) && is(get(n.k), K::FullK, *n.layer_id) &&
               is(get(n.v), K::V, *n.layer_id) && is(get(n.position), K::CachePositions),
               "attention Q/K/V/position/scale/dtype differs");
          put(n.out, {K::AttentionHeads, *n.layer_id}); break;
        }
        default: need(false, "operation outside explicit GPT-NeoX adapter");
      }
    }
    need(used_norms.size() == norms.size() && used_projections.size() == projections.size() &&
         used_attention.size() == size_t(config_.layers) && used_split.size() == size_t(config_.layers) &&
         used_gelu.size() == size_t(config_.layers) && cosine_count == 1 && sine_count == 1 &&
         frequency_matmuls == 1 && head_count == 1 && embedding_count == 1 &&
         is(get(Tid{p.output_map[0].idx}), K::LogitsFloat), "incomplete native GPT-NeoX graph");
  }
};
} // namespace executorch::backends::mlx
