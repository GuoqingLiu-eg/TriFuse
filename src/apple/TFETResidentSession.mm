#include <executorch/extension/module/module.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/cache/attention_selection.h>
#include <executorch/backends/mlx/runtime/backend_options.h>
#include <executorch/backends/mlx/runtime/MLXSequenceCache.h>
#ifdef TRIFUSE_ENABLE_COLD_KV
#include <trifuse/kv/ColdKVInstaller.h>
#endif
#include <executorch/backends/mlx/runtime/MLXConstantPool.h>
#include <executorch/backends/mlx/runtime/resident_prefill.h>
#include <executorch/backends/mlx/runtime/resident_features.h>
#include <executorch/backends/mlx/runtime/resident_attention_labels.h>
#include <executorch/backends/mlx/runtime/resident_greedy.h>
#include <executorch/runtime/platform/runtime.h>
#include <mlx/memory.h>
#include <mlx/device.h>
#include <mlx/utils.h>
#include <mlx/backend/metal/metal.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#import "TFETResidentSession.h"
#include "TFETSharedSegmentDataLoader.h"

namespace {
using namespace executorch::extension;
using namespace executorch::runtime;
using executorch::aten::ScalarType;
namespace cache = executorch::extension::llm::cache;
namespace backend = executorch::backends::mlx;
void need(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
struct NativeExecutionCancelled final : std::runtime_error {
  NativeExecutionCancelled() : std::runtime_error("Native execution permission cancelled") {}
};
using PrefillTimingClock = std::chrono::steady_clock;
bool prefillTimingEnabled() {
  const char *value = std::getenv("TRIFUSE_PREFILL_TIMING");
  return value && std::strcmp(value, "1") == 0;
}
int prefillMaxQueuedChunks() {
  const char *value = std::getenv("TRIFUSE_PREFILL_MAX_QUEUED_CHUNKS");
  if (!value) return 4;
  need(*value != '\0', "TRIFUSE_PREFILL_MAX_QUEUED_CHUNKS must be an integer1..64");
  int count = 0;
  for (const char *p = value; *p; ++p) {
    need(*p >= '0' && *p <= '9', "TRIFUSE_PREFILL_MAX_QUEUED_CHUNKS must be an integer1..64");
    count = count * 10 + (*p - '0');
    need(count <= 64, "TRIFUSE_PREFILL_MAX_QUEUED_CHUNKS must be an integer1..64");
  }
  need(count >= 1, "TRIFUSE_PREFILL_MAX_QUEUED_CHUNKS must be an integer1..64");
  return count;
}
void prefillTiming(const char *stage, PrefillTimingClock::time_point start,
                  size_t chunk, int position, size_t tokens, bool native, bool last,
                  int max_queued_chunks) {
  const double wall_ms = std::chrono::duration<double, std::milli>(
      PrefillTimingClock::now() - start).count();
  // Host wall time only. Forward of a cache-only chunk is an async submission.
  std::fprintf(stderr,
      "TRIFUSE_PREFILL_TIMING bridge stage=%s chunk=%zu position=%d "
      "tokens=%zu native=%d last=%d max_queued_chunks=%d wall_ms=%.6f\n",
      stage, chunk, position, tokens, int(native), int(last), max_queued_chunks, wall_ms);
  std::fflush(stderr);
}
void diagnosticMemory(const char *phase) {
  const char *enabled = std::getenv("TRIFUSE_RESIDENT_PROGRESS");
  if (!enabled || std::strcmp(enabled, "1")) return;
  std::printf("TRIFUSE_RESIDENT_NATIVE %s active_bytes=%zu cached_bytes=%zu peak_active_bytes=%zu\n",
      phase, mlx::core::get_active_memory(), mlx::core::get_cache_memory(), mlx::core::get_peak_memory());
  std::fflush(stdout);
}
int64_t exactInteger(id value) {
  need([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(), "Integer NSNumber required");
  const char *type = [(NSNumber *)value objCType];
  need(type && std::strchr("csilqCSILQ", type[0]) && type[1] == '\0', "Fractional/floating NSNumber is not an integer input");
  if (std::strchr("CSILQ", type[0])) need([(NSNumber *)value unsignedLongLongValue] <= uint64_t(INT64_MAX), "Integer outside signed64 range");
  return [(NSNumber *)value longLongValue];
}
backend::NativeLlamaPrefillConfig parseNativeConfiguration(NSDictionary<NSString *, id> *configuration) {
  need([configuration isKindOfClass:NSDictionary.class], "Explicit native architecture configuration required");
  const bool neox = [configuration[@"architecture"] isEqual:@"gpt_neox_parallel_v1"];
  const bool granite = [configuration[@"architecture"] isEqual:@"granite_moe_v1"];
  const bool olmoe = [configuration[@"architecture"] isEqual:@"olmoe_v1"];
  need(neox || granite || olmoe || [configuration[@"architecture"] isEqual:@"standard_llama_tied_v1"],
       "Unsupported native architecture");
  auto integer = [&](NSString *key) -> int {
    const auto value = exactInteger(configuration[key]);
    need(value > 0 && value <= INT32_MAX, "Native adapter integer outside range");
    return int(value);
  };
  auto real = [&](NSString *key) -> float {
    id value = configuration[key];
    need([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(), "Native adapter finite number required");
    const float result = [value floatValue];
    need(std::isfinite(result) && result > 0, "Native adapter real outside range");
    return result;
  };
  backend::NativeLlamaPrefillConfig config;
  config.layers = integer(@"layers"); config.hidden_size = integer(@"hidden_size");
  config.query_heads = integer(@"query_heads"); config.kv_heads = integer(@"kv_heads");
  config.head_dim = integer(@"head_dim"); config.intermediate_size = integer(@"intermediate_size");
  config.vocabulary = integer(@"vocabulary"); config.capacity = integer(@"capacity");
  config.rope_theta = real(@"rope_theta");
  if (neox) {
    config.architecture = backend::NativeModelArchitecture::GPTNeoXParallel;
    config.layer_norm_eps = real(@"layer_norm_eps");
    config.rotary_dimensions = integer(@"rotary_dimensions");
    need(!configuration[@"rms_norm_eps"], "GPT-NeoX uses LayerNorm, not RMSNorm");
  } else {
    config.rms_norm_eps = real(@"rms_norm_eps");
    need(!configuration[@"layer_norm_eps"] && !configuration[@"rotary_dimensions"],
         "RMSNorm adapters cannot use GPT-NeoX normalization/rotary fields");
  }
  if (granite || olmoe) {
    config.architecture = olmoe ? backend::NativeModelArchitecture::OLMoE : backend::NativeModelArchitecture::GraniteMoE;
    need([configuration[@"quantization_layout"] isEqual:@"affine_per_projection"],
         "Granite MoE requires actual per-projection affine quantization");
    config.quantization_layout = backend::NativeQuantizationLayout::AffinePerProjection;
    config.num_experts = integer(@"num_experts");
    config.experts_per_token = integer(@"experts_per_token");
    config.embedding_multiplier = real(@"embedding_multiplier");
    config.residual_multiplier = real(@"residual_multiplier");
    config.attention_multiplier = real(@"attention_multiplier");
    config.logits_scaling = real(@"logits_scaling");
    id tied = configuration[@"tie_word_embeddings"];
    need([tied isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)tied) == CFBooleanGetTypeID(),
         "Granite tied embedding declaration must be a boolean");
    config.tie_word_embeddings = [tied boolValue];
    if (olmoe) {
      id renormalize = configuration[@"norm_topk_prob"];
      need([renormalize isKindOfClass:NSNumber.class] &&
           CFGetTypeID((__bridge CFTypeRef)renormalize) == CFBooleanGetTypeID() && ![renormalize boolValue],
           "OLMoE requires original norm_topk_prob=false");
      need([configuration[@"qk_norm"] isEqual:@"full_projection_width"],
           "OLMoE requires Q/K RMSNorm over each complete projection before head splitting");
    }
  } else {
    need(!configuration[@"quantization_layout"] || [configuration[@"quantization_layout"] isEqual:@"uniform"],
         "Legacy native adapters require uniform quantization");
  }
  if (configuration[@"linear_quantization_bits"] || configuration[@"linear_quantization_group_size"]) {
    const auto bits = exactInteger(configuration[@"linear_quantization_bits"]);
    const auto group = exactInteger(configuration[@"linear_quantization_group_size"]);
    need((bits == 0 && group == 0) || (!granite && !olmoe && bits == 8 && group == 32),
         "Native uniform quantization must be dense 0/0 or affine 8/32; mixed layouts use node metadata");
    config.linear_quantization_bits = int(bits);
    config.linear_quantization_group_size = int(group);
  }
  backend::validate_native_model_configuration(config);
  return config;
}
bool sameNativeConfiguration(const backend::NativeLlamaPrefillConfig& a,
                             const backend::NativeLlamaPrefillConfig& b) {
  return a.architecture == b.architecture && a.rotary_dimensions == b.rotary_dimensions &&
      a.quantization_layout == b.quantization_layout &&
      a.num_experts == b.num_experts && a.experts_per_token == b.experts_per_token &&
      a.embedding_multiplier == b.embedding_multiplier && a.residual_multiplier == b.residual_multiplier &&
      a.attention_multiplier == b.attention_multiplier && a.logits_scaling == b.logits_scaling &&
      a.tie_word_embeddings == b.tie_word_embeddings &&
      a.layer_norm_eps == b.layer_norm_eps &&
      a.layers == b.layers && a.query_heads == b.query_heads && a.kv_heads == b.kv_heads &&
      a.head_dim == b.head_dim && a.hidden_size == b.hidden_size && a.intermediate_size == b.intermediate_size &&
      a.vocabulary == b.vocabulary && a.capacity == b.capacity &&
      a.rms_norm_eps == b.rms_norm_eps && a.rope_theta == b.rope_theta &&
      a.linear_quantization_bits == b.linear_quantization_bits &&
      a.linear_quantization_group_size == b.linear_quantization_group_size;
}
struct Invocation {
  std::vector<int64_t> tokens, positions;
  std::vector<TensorPtr> tensors;
  std::vector<EValue> values;
};

// This queue owns only summary tickets and input leases. It never reads or
// mutates a cache owner, so waiting here cannot invert the owner's lock order.
struct FeatureObserver : std::enable_shared_from_this<FeatureObserver> {
  backend::ResidentFeatureConfig config;
  size_t limit;
  dispatch_queue_t queue = dispatch_queue_create("trifuse.prefill.features", DISPATCH_QUEUE_SERIAL);
  dispatch_group_t group = dispatch_group_create();
  void (^consumer)(NSDictionary<NSString *, id> *);
  NSString *session, *weights;
  std::mutex mutex;
  size_t pending = 0, peak = 0, attempted = 0, skipped = 0, completed = 0, failed = 0, bytes = 0;
  double readMS = 0, consumeMS = 0;
  NSMutableArray<NSDictionary *> *records = [NSMutableArray array];
  FeatureObserver(backend::ResidentFeatureConfig c, size_t maximum,
                  void (^callback)(NSDictionary<NSString *, id> *), NSString *sid, NSString *wid)
      : config(std::move(c)), limit(maximum), consumer([callback copy]),
        session([sid copy]), weights([wid copy]) {}
  bool reserve() {
    std::lock_guard<std::mutex> lock(mutex);
    ++attempted;
    if (pending >= limit) { ++skipped; return false; }
    ++pending; peak = std::max(peak, pending);
    dispatch_group_enter(group);
    return true;
  }
  void cancelReservation() {
    std::lock_guard<std::mutex> lock(mutex);
    --pending;
    dispatch_group_leave(group);
  }
  void submit(std::shared_ptr<backend::FeatureTicket> ticket,
              std::shared_ptr<Invocation> inputLease, uint64_t generation) {
    auto self = shared_from_this();
    // Allocate before queuing. Unknown producer completion retains this lease
    // until process exit, matching the native execution failure policy.
    auto *retainedInput = new std::shared_ptr<Invocation>(std::move(inputLease));
    dispatch_async(queue, ^{
      @autoreleasepool {
        const auto start = PrefillTimingClock::now();
        double readTime = 0, consumeTime = 0;
        size_t ownedBytes = 0;
        NSMutableDictionary *record = [NSMutableDictionary dictionary];
        try {
          auto batch = ticket->wait_and_read();
          readTime = std::chrono::duration<double, std::milli>(PrefillTimingClock::now() - start).count();
          ownedBytes = batch.values.size() * sizeof(float);
          NSMutableArray *layers = [NSMutableArray arrayWithCapacity:batch.layers.size()];
          for (int layer : batch.layers) [layers addObject:@(layer)];
          NSData *values = [NSData dataWithBytes:batch.values.data() length:ownedBytes];
          need(values != nil, "Could not copy prefill feature values");
          NSDictionary *payload = @{
            @"schema":@"prefill_new_tensor_stats_v1", @"session_id":self->session,
            @"effective_weights_id":self->weights, @"generation_before_batch":@(generation),
            @"expected_generation_after_batch":@(generation + 1),
            @"state_scope":@"provisional_chunk_observation_not_a_cache_commit",
            @"start":@(batch.absolute_start), @"count":@(batch.token_count),
            @"layers":layers, @"block_tokens":@(batch.block_tokens),
            @"feature_names":@[@"h_mean_square",@"h_max_abs",@"k_mean_square",@"k_max_abs",@"v_mean_square",@"v_max_abs"],
            @"feature_scope":@"input_RMSNorm_hidden_and_new_preRoPE_K_V_only",
            @"values":values, @"advisory_only":@YES};
          record[@"start"] = payload[@"start"]; record[@"count"] = payload[@"count"];
          record[@"bytes"] = @(ownedBytes);
          for (NSString *key in @[@"generation_before_batch", @"expected_generation_after_batch", @"state_scope", @"feature_scope"])
            record[key] = payload[key];
          const auto consumeStart = PrefillTimingClock::now();
          self->consumer(payload);
          consumeTime = std::chrono::duration<double, std::milli>(PrefillTimingClock::now() - consumeStart).count();
          record[@"completed"] = @YES;
        } catch (const std::exception& e) {
          record[@"completed"] = @NO;
          record[@"error"] = [NSString stringWithUTF8String:e.what()];
        } catch (...) {
          record[@"completed"] = @NO; record[@"error"] = @"Unknown feature observation error";
        }
        // Release the input only after the summary's producer completion was
        // observed. Main prefill independently retains all its input leases.
        if ([record[@"completed"] boolValue]) delete retainedInput;
        record[@"wait_and_read_ms"] = @(readTime);
        record[@"consumer_ms"] = @(consumeTime);
        {
          std::lock_guard<std::mutex> lock(self->mutex);
          if ([record[@"completed"] boolValue]) ++self->completed; else ++self->failed;
          self->bytes += ownedBytes; self->readMS += readTime; self->consumeMS += consumeTime;
          [self->records addObject:record]; --self->pending;
        }
        dispatch_group_leave(self->group);
      }
    });
  }
  NSDictionary *drain() {
    const auto start = PrefillTimingClock::now();
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    const double waitMS = std::chrono::duration<double, std::milli>(PrefillTimingClock::now() - start).count();
    std::lock_guard<std::mutex> lock(mutex);
    return @{@"schema":@"prefill_feature_observer_v1", @"attempted_batches":@(attempted),
      @"skipped_before_collection":@(skipped), @"completed_batches":@(completed), @"failed_batches":@(failed),
      @"pending_batches":@(pending), @"max_pending":@(limit), @"peak_pending":@(peak),
      @"host_feature_bytes":@(bytes), @"wait_and_read_sum_ms":@(readMS),
      @"consumer_sum_ms":@(consumeMS), @"final_drain_wait_ms":@(waitMS),
      @"records":[records copy], @"advisory_only":@YES,
      @"scope":@"No cache action. Host timings do not prove ANE placement or GPU overlap; all summary and callback cost remains inside the complete request."};
  }
};

struct FeatureReservation {
  std::shared_ptr<FeatureObserver> observer;
  bool reserved = false, submitted = false;
  explicit FeatureReservation(std::shared_ptr<FeatureObserver> value) : observer(std::move(value)) {}
  ~FeatureReservation() { if (reserved && !submitted) observer->cancelReservation(); }
};
struct GreedyPipeline {
  backend::NativeLlamaPrefillConfig config;
  uint64_t generation = 0;
  int prompt_position = 0, first_token = 0, submitted = 0, returned = 0, completed = 0;
  bool clear_due = false;
  int pending_prompt_tokens = 0;
  bool prompt_submitted = false;
  int decode_clears = 0;
  std::shared_ptr<backend::ResidentMoEPolicyContext> policy;
  std::shared_ptr<backend::ResidentGreedyTicket> prediction, submitted_prediction;
  std::shared_ptr<Invocation> active;
};
std::unique_ptr<DataLoader> residentProgramLoader(const char *path) {
  auto loader = MmapDataLoader::from(path, MmapDataLoader::MlockConfig::NoMlock);
  need(loader.ok(), "Could not open resident program mapping");
  return std::make_unique<MmapDataLoader>(std::move(loader.get()));
}
std::shared_ptr<TFETSharedSegmentDataLoader::State> residentWeightMappings(const char *path) {
  if (!path) return nullptr;
  auto state = TFETSharedSegmentDataLoader::createState(path);
  need(state.ok(), "Could not open shared resident weight mapping");
  return std::move(state.get());
}
std::unique_ptr<DataLoader> residentWeightLoader(
    const std::shared_ptr<TFETSharedSegmentDataLoader::State>& state) {
  return state ? std::make_unique<TFETSharedSegmentDataLoader>(state) : nullptr;
}
std::shared_ptr<backend::MLXConstantPool> residentConstantPool() {
  const auto mode = backend::parse_mlx_constant_pool_mode(
      std::getenv("TRIFUSE_MLX_CONSTANT_POOL"));
  if (mode == backend::MLXConstantPoolMode::Off) return nullptr;
  return std::make_shared<backend::MLXConstantPool>(mode);
}
struct Core {
#ifdef TRIFUSE_ENABLE_COLD_KV
  std::unique_ptr<ColdKVInstaller> coldInstaller;
#endif
  // Reverse destruction: invocation, Modules, options, registry session, cache.
  std::shared_ptr<backend::MLXSequenceCache> storage;
  cache::CacheSession binding;
  std::string cpuBLASThreading;
  BackendOptions<2> options;
  BackendOptions<3> decodeOptions;
  LoadBackendOptionsMap optionsMap, decodeOptionsMap;
  // Prefill/decode/verify use the same immutable PTD segments. Each returned
  // buffer owns a lease, so source mappings also survive quarantine/teardown.
  std::shared_ptr<TFETSharedSegmentDataLoader::State> weightMappings;
  // Owner-local immutable wrappers, destroyed after Modules. Canonical array
  // storage independently owns its mapping lease if a graph outlives the Core.
  std::shared_ptr<backend::MLXConstantPool> constantPool;
  Module prefill, decode;
  std::unique_ptr<Module> verify;
  std::unique_ptr<Invocation> active;
  std::vector<std::shared_ptr<Invocation>> prefillBatch;
  std::optional<backend::NativeLlamaPrefillConfig> nativePrefillConfiguration;
  uint64_t nativePrefillGeneration = 0;
  // An import source remains owned here until complete or quarantined.
  std::unique_ptr<backend::FlatFP32State> importing;
  std::unique_ptr<GreedyPipeline> greedyPipeline;
  std::shared_ptr<backend::ResidentMoEPolicyContext> policyContext;
  unsigned policyContextCreations = 0;
  void ensurePolicyContext() {
    // beginRequest resets storage contents, not its identity. MLX streams live
    // in a global registry, so keep one policy stream for this resident owner.
    if (!policyContext) {
      policyContext = std::make_shared<backend::ResidentMoEPolicyContext>(storage.get());
      ++policyContextCreations;
    }
  }
  Core(const cache::CacheConfig& config, const char *p, const char *d, const char *v, const char *data, bool decodeOnCPU, const char *blasPolicy)
    : storage(std::make_shared<backend::MLXSequenceCache>(config)),
      binding(cache::make_unique_key(), storage),
      cpuBLASThreading(blasPolicy ? blasPolicy : "default"),
      weightMappings(residentWeightMappings(data)),
      constantPool(residentConstantPool()),
      prefill(residentProgramLoader(p), nullptr, nullptr, nullptr, residentWeightLoader(weightMappings)),
      decode(residentProgramLoader(d), nullptr, nullptr, nullptr, residentWeightLoader(weightMappings)),
      verify(v ? std::make_unique<Module>(residentProgramLoader(v), nullptr, nullptr, nullptr,
                                        residentWeightLoader(weightMappings)) : nullptr) {
#ifdef TRIFUSE_ENABLE_COLD_KV
    coldInstaller = std::make_unique<ColdKVInstaller>(config.n_layers);
#endif
    options.set_option(backend::kCacheKeyKey, binding.key().c_str());
    options.set_option(backend::kDeviceKey, "gpu");
    optionsMap.set_options(backend::kMLXBackendId, options.view());
    decodeOptions.set_option(backend::kCacheKeyKey, binding.key().c_str());
    decodeOptions.set_option(backend::kDeviceKey, decodeOnCPU ? "cpu" : "gpu");
    if (decodeOnCPU && blasPolicy) decodeOptions.set_option(backend::kCPUBlasThreadingKey, cpuBLASThreading.c_str());
    decodeOptionsMap.set_options(backend::kMLXBackendId, decodeOptions.view());
  }
};
void quarantine(std::unique_ptr<Core> context) noexcept {
  // Intentional process-lifetime retention. Do not allocate a bookkeeping
  // container on an error path: allocation failure must not destroy this Core.
  (void)context.release();
}
int64_t constant(Module& module, const char *name) {
  const auto value = module.execute(name);
  need(value.ok() && value->size() == 1 && value->at(0).isInt(), "Missing or invalid resident PTE scalar metadata");
  return value->at(0).toInt();
}
void constantVector(Module& module, const char *name, int layers, int expected) {
  const auto value = module.execute(name);
  need(value.ok() && value->size() == 1 && value->at(0).isTensor(), "Missing resident PTE vector metadata");
  const auto& tensor = value->at(0).toTensor();
  need(tensor.scalar_type() == ScalarType::Int && tensor.dim() == 1 && tensor.size(0) == layers && tensor.strides()[0] == 1 && tensor.const_data_ptr<int32_t>(), "Wrong resident PTE vector metadata ABI");
  for (int i = 0; i < layers; ++i) need(tensor.const_data_ptr<int32_t>()[i] == expected, "Resident PTE cache geometry differs");
}
void validateModule(Module& module, Core& core, int block, int outputTokens, int layers, int heads, int dim, int capacity, int vocab, int kvScalar, int prefillTokens, TFETNativeExecutionPermit *permit) {
  need(module.load() == Error::Ok, "Resident PTE load failed");
  need(constant(module, "get_n_caches") == layers && constant(module, "get_max_ctx_len") == capacity && constant(module, "get_prefill_chunk_size") == prefillTokens, "Resident PTE cache limits/prefill shape differ");
  constantVector(module, "get_kv_heads", layers, heads);
  constantVector(module, "get_head_dims", layers, dim);
  constantVector(module, "get_windows", layers, 0);
  const auto names = module.method_names();
  need(names.ok(), "Could not read compiled resident methods");
  if (names->count("get_kv_dtype")) {
    need(constant(module, "get_kv_dtype") == kvScalar, "Compiled cache precision differs from requested storage");
  } else {
    need(kvScalar == int(ScalarType::Float), "FP16 cache requires explicit compiled get_kv_dtype metadata");
  }
  if (permit.isCancelled) throw NativeExecutionCancelled();
  auto& loadOptions = &module == &core.decode ? core.decodeOptionsMap : core.optionsMap;
  const auto bindingError = [&] {
    // Delegate initialization is synchronous. Scope only this owner's load,
    // restoring any outer scope on success or exception; never crosses threads.
    backend::MLXConstantPoolLoadScope constantsScope(core.constantPool);
    return module.load_method("forward", nullptr, nullptr, &loadOptions);
  }();
  if (bindingError != Error::Ok) throw std::runtime_error(
      "Resident forward/cache binding failed (ExecuTorch error " +
      std::to_string(static_cast<uint32_t>(bindingError)) + ")");
  const auto meta = module.method_meta("forward");
  need(meta.ok() && meta->num_inputs() == 2 && meta->num_outputs() == 1, "Resident method arity differs");
  const std::vector<std::vector<int32_t>> shapes{{1, block}, {block}, {1, outputTokens, vocab}};
  for (int i = 0; i < 3; ++i) {
    const auto t = i < 2 ? meta->input_tensor_meta(i) : meta->output_tensor_meta(0);
    need(t.ok() && t->scalar_type() == (i < 2 ? ScalarType::Long : ScalarType::Float) && t->sizes().size() == shapes[i].size(), "Resident method dtype/rank differs");
    for (size_t j = 0; j < shapes[i].size(); ++j) need(t->sizes()[j] == shapes[i][j] && t->dim_order()[j] == j, "Resident method shape/order differs");
  }
}
}

@interface TFETAttentionSelectionChannel ()
- (instancetype)initWithMailbox:(std::shared_ptr<cache::AttentionSelectionMailbox>)mailbox
                       capacity:(int)capacity heads:(int)heads;
@end
@implementation TFETAttentionSelectionChannel {
  std::shared_ptr<cache::AttentionSelectionMailbox> _mailbox;
  int _capacity, _heads;
}
- (instancetype)initWithMailbox:(std::shared_ptr<cache::AttentionSelectionMailbox>)mailbox
                       capacity:(int)capacity heads:(int)heads {
  if ((self = [super init])) { _mailbox = std::move(mailbox); _capacity = capacity; _heads = heads; }
  return self;
}
- (uint64_t)requestEpoch { return _mailbox->epoch(); }
- (nullable NSNumber *)publishPositionsByHead:(NSArray<NSArray<NSNumber *> *> *)positions
    coveredLength:(NSInteger)coveredLength validUntilPosition:(NSInteger)validUntilPosition
    requestEpoch:(uint64_t)requestEpoch proposalID:(uint64_t)proposalID error:(NSError **)error {
  try {
    need(coveredLength > 0 && coveredLength <= _capacity &&
         validUntilPosition >= coveredLength && validUntilPosition <= _capacity,
         "Selection coverage/deadline outside capacity");
    need([positions isKindOfClass:NSArray.class] && positions.count == size_t(_heads),
         "Selection must specify each KV head");
    auto proposal = std::make_shared<cache::AttentionSelection>();
    proposal->request_epoch = requestEpoch; proposal->proposal_id = proposalID;
    proposal->covered_length = int(coveredLength); proposal->valid_until_position = int(validUntilPosition);
    for (NSArray<NSNumber *> *head in positions) {
      need([head isKindOfClass:NSArray.class] && head.count > 0 && head.count <= size_t(coveredLength),
           "Selection head size outside covered prefix");
      std::vector<int32_t> indices; indices.reserve(head.count);
      for (NSNumber *item in head) {
        const auto value = exactInteger(item);
        need(value >= 0 && value < coveredLength, "Selection position outside covered prefix");
        indices.push_back(int32_t(value));
      }
      proposal->positions_by_head.push_back(std::move(indices));
    }
    return @(_mailbox->publish(std::move(proposal)));
  } catch (const std::exception& e) {
    if (error) *error = [NSError errorWithDomain:@"TriFuse.Selection" code:1 userInfo:@{
      NSLocalizedDescriptionKey:[NSString stringWithUTF8String:e.what()]}];
    return nil;
  }
}
@end

@implementation TFETNativeExecutionPermit {
  std::atomic<bool> _cancelled;
}
- (instancetype)init {
  self = [super init];
  if (self) _cancelled.store(false, std::memory_order_relaxed);
  return self;
}
- (void)cancel { _cancelled.store(true, std::memory_order_release); }
- (BOOL)isCancelled { return _cancelled.load(std::memory_order_acquire); }
@end

@implementation TFETResidentSession {
  std::mutex _mutex;
  std::unique_ptr<Core> _core;
  TFETNativeExecutionPermit *_nativeExecutionPermit;
  NSString *_nativeCancellationCompletion;
  BOOL _nativeCancellationKnownIdle;
  int _layers, _heads, _dim, _capacity, _vocabulary, _position, _verifyTokens, _cacheWindow, _physicalCapacity, _kvScalar, _kvElementBytes, _prefillTokens, _prefillMaxQueuedChunks;
  size_t _elements;
  uint64_t _generation;
  BOOL _invalid, _decodeOnCPU, _retainedCompacted, _requestBounded;
  NSNumber *_kvPayloadBudgetBytes;
  backend::RetainedStorageStats _retainedStats;
  NSString *_weights;
  NSString *_sessionID;
  NSString *_kvDtype, *_cpuBLASThreading;
  std::shared_ptr<FeatureObserver> _features;
  TFETAttentionSelectionChannel *_selectionChannel;
  std::shared_ptr<cache::AttentionSelectionMailbox> _selectionMailbox;
  std::shared_ptr<const cache::AttentionSelection> _activeSelection;
  uint64_t _selectionSteps, _selectionFullSteps;
  BOOL _selectionUsed;
  NSMutableDictionary<NSString *, id> *_greedyPipelineReceipt;
}
- (void)dealloc {
  // nextGreedyPipeline may return while a consuming T1 still uses mapped
  // weights and invocation buffers. An abandoned owner must not destroy them
  // until submitted work finishes. Never evaluate new lazy roots here.
  if (_core && _core->greedyPipeline) {
    try {
      backend::drain_submitted_work();
      _core.reset();
    } catch (...) {
      // Destruction cannot report a drain failure. Preserve the complete
      // ownership chain if completion is unknown instead of freeing in-use
      // memory. Ordinary finished owners take the existing destruction path.
      quarantine(std::move(_core));
    }
  }
}
- (nullable TFETAttentionSelectionChannel *)enableAttentionSelectionExpectedGeneration:(uint64_t)generation error:(NSError **)error {
  return [self enableAttentionSelectionReuseHotViews:NO expectedGeneration:generation error:error];
}
- (nullable TFETAttentionSelectionChannel *)enableAttentionSelectionReuseHotViews:(BOOL)reuseHotViews
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  return [self enableAttentionSelectionReuseHotViews:reuseHotViews deltaReuse:NO expectedGeneration:generation error:error];
}
- (nullable TFETAttentionSelectionChannel *)enableAttentionSelectionReuseHotViews:(BOOL)reuseHotViews
    deltaReuse:(BOOL)deltaReuse expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex);
  try {
    [self admit:generation];
    need(!_nativeExecutionPermit, "This diagnostic API is unavailable for a permission-bound owner");
    need(!deltaReuse || reuseHotViews, "Delta reuse requires hot-view reuse");
    need(!_selectionChannel && _position == 0 && !_decodeOnCPU && !_cacheWindow && !_core->verify &&
         !_retainedCompacted && _core->storage->has_full_history_storage(),
         "Attention selection requires a fresh full-history GPU target without verifier");
    _selectionMailbox = std::make_shared<cache::AttentionSelectionMailbox>(_capacity, _heads);
    _selectionMailbox->begin_request();
    _core->storage->set_attention_selection_reuse(bool(reuseHotViews));
    _core->storage->set_attention_selection_delta_reuse(bool(deltaReuse));
    _selectionChannel = [[TFETAttentionSelectionChannel alloc] initWithMailbox:_selectionMailbox capacity:_capacity heads:_heads];
    return _selectionChannel;
  } catch (const std::exception& e) {
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:NO]; return nil;
  }
}
- (void)resetAttentionSelection {
  if (!_selectionMailbox) return;
  _selectionMailbox->begin_request();
  _activeSelection.reset();
  _core->storage->set_attention_selection(nullptr);
}
+ (BOOL)cpuDecodeAvailable {
  return mlx::core::is_available(mlx::core::Device(mlx::core::Device::cpu));
}
+ (BOOL)configureMetallibPath:(NSString *)path error:(NSError **)error {
  try {
    need(path.length && [path hasSuffix:@".metallib"] &&
         [[NSFileManager defaultManager] fileExistsAtPath:path], "Explicit metallib file required");
    mlx::core::metal::set_metallib_path(path.fileSystemRepresentation);
    return YES;
  } catch (const std::exception& e) {
    if (error) *error = [NSError errorWithDomain:@"TriFuse.Runtime" code:1 userInfo:@{
      NSLocalizedDescriptionKey:[NSString stringWithUTF8String:e.what()]}];
    return NO;
  }
}
+ (NSDictionary<NSString *, id> *)runtimeConfiguration {
  return @{@"effectiveTF32":@(mlx::core::env::enable_tf32()),
    @"explicitMetallibPath":[NSString stringWithUTF8String:mlx::core::metal::get_metallib_path().c_str()]};
}
+ (nullable NSDictionary<NSString *, id> *)restoreIdleCacheLimitBytes:(NSNumber *)limit error:(NSError **)error {
  const auto started = PrefillTimingClock::now();
  try {
    const auto value = exactInteger(limit);
    need(value >= 0 && uint64_t(value) <= std::numeric_limits<size_t>::max(), "Idle cache limit must be a nonnegative byte count");
    const auto previous = backend::restore_idle_cache_limit(size_t(value));
    const double elapsed = std::chrono::duration<double, std::milli>(PrefillTimingClock::now() - started).count();
    return @{@"schema":@"idle_cache_limit_restoration_v1", @"restoredCacheLimitBytes":@(value),
      @"replacedCacheLimitBytes":@(previous), @"wallMS":@(elapsed),
      @"scope":@"Process-wide limit setter only; no model access, GPU drain or cache clearing."};
  } catch (const std::exception& e) {
    if (error) *error = [NSError errorWithDomain:@"TriFuse.Runtime" code:1 userInfo:@{
      NSLocalizedDescriptionKey:[NSString stringWithUTF8String:e.what()]}];
    return nil;
  } catch (...) {
    if (error) *error = [NSError errorWithDomain:@"TriFuse.Runtime" code:1 userInfo:@{
      NSLocalizedDescriptionKey:@"Idle cache limit restoration failed"}];
    return nil;
  }
}
- (NSArray<NSNumber *> *)shape { return @[@(_layers), @1, @(_heads), @(_physicalCapacity), @(_dim)]; }
- (void)setError:(NSError **)error message:(NSString *)message dispatched:(BOOL)dispatched {
  if (error) *error = [NSError errorWithDomain:@"TriFuse.Resident" code:dispatched ? 2 : 1 userInfo:@{
    NSLocalizedDescriptionKey:message, @"completion":dispatched ? @"unknown" : @"not_dispatched",
    @"sessionInvalid":@(_invalid), @"committedPosition":@(_position), @"generation":@(_generation)}];
}
- (void)admit:(uint64_t)generation {
  need(!_invalid && bool(_core), "Resident session is invalid");
  need(!_core->greedyPipeline, "Resident owner has an active greedy pipeline");
  need(generation == _generation, "Stale resident generation");
  need(_generation < uint64_t(INT64_MAX), "Resident generation exhausted");
}
- (void)checkNativeExecutionPermit {
  if (_nativeExecutionPermit.isCancelled) throw NativeExecutionCancelled();
}
- (void)finishNativeCancellation:(NSError **)error dispatched:(BOOL)dispatched {
  // Never evaluate lazy KV roots here: that could launch new model work after
  // cancellation. Keep all input leases alive until submitted work is drained.
  BOOL drained = NO;
  _nativeCancellationKnownIdle = NO;
  NSString *drainFailure = nil;
  if (dispatched) {
    _invalid = YES;
    try {
      backend::drain_submitted_work();
      _core.reset();
      drained = YES;
    } catch (const std::exception& e) {
      drainFailure = [NSString stringWithUTF8String:e.what()];
      [self failDispatched];
    } catch (...) {
      drainFailure = @"Unknown submitted-work drain failure";
      [self failDispatched];
    }
  } else if (_core) {
    // Batch inputs may have been staged, but no operation was admitted.
    _core->prefillBatch.clear();
    // dispatched=false proves forward was never entered. A staged first
    // invocation/ticket has no GPU work and can be released without quarantine.
    if (_core->greedyPipeline && _core->greedyPipeline->submitted == 0 && !_core->greedyPipeline->prompt_submitted)
      _core->greedyPipeline.reset();
    _nativeCancellationKnownIdle = !_core->active && !_core->importing &&
        !_core->greedyPipeline &&
        _core->storage->length() == _position;
  }
  _nativeCancellationCompletion = !dispatched ? @"not_dispatched" :
      (drained ? @"cancelled_drained" : @"unknown");
  if (_greedyPipelineReceipt) {
    _greedyPipelineReceipt[@"state"] = @"cancelled";
    _greedyPipelineReceipt[@"cancellationCompletion"] = _nativeCancellationCompletion;
    _greedyPipelineReceipt[@"committedPosition"] = @(_position);
  }
  if (error) *error = [NSError errorWithDomain:@"TriFuse.Resident" code:3 userInfo:@{
    NSLocalizedDescriptionKey:@"Native execution permission cancelled",
    @"cancelled":@YES, @"nativeExecutionCancelled":@YES, @"completion":_nativeCancellationCompletion,
    @"cancellationDrained":@(drained), @"drainFailure":drainFailure ?: NSNull.null,
    @"sessionInvalid":@(_invalid), @"committedPosition":@(_position), @"generation":@(_generation)}];
}
- (void)failDispatched {
  _invalid = YES;
  if (_greedyPipelineReceipt && _core && _core->greedyPipeline)
    _greedyPipelineReceipt[@"state"] = @"invalid";
  if (_selectionMailbox) _selectionMailbox->begin_request();
  _activeSelection.reset();
  if (_core) quarantine(std::move(_core));
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:nil verifyTokens:0
      layers:layers kvHeads:heads headDim:dim capacity:capacity vocabulary:vocabulary effectiveWeightsID:weights error:error];
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens
    layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:verifyPath verifyTokens:verifyTokens
      dataPath:nil layers:layers kvHeads:heads headDim:dim capacity:capacity vocabulary:vocabulary effectiveWeightsID:weights error:error];
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens dataPath:(nullable NSString *)dataPath
    layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:verifyPath verifyTokens:verifyTokens
      dataPath:dataPath cacheWindow:0 layers:layers kvHeads:heads headDim:dim capacity:capacity vocabulary:vocabulary effectiveWeightsID:weights error:error];
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens dataPath:(nullable NSString *)dataPath cacheWindow:(NSInteger)cacheWindow
    layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:verifyPath verifyTokens:verifyTokens
      dataPath:dataPath cacheWindow:cacheWindow decodeOnCPU:NO layers:layers kvHeads:heads headDim:dim capacity:capacity
      vocabulary:vocabulary effectiveWeightsID:weights error:error];
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens dataPath:(nullable NSString *)dataPath cacheWindow:(NSInteger)cacheWindow
    decodeOnCPU:(BOOL)decodeOnCPU layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:verifyPath verifyTokens:verifyTokens
      dataPath:dataPath cacheWindow:cacheWindow decodeOnCPU:decodeOnCPU kvDtype:@"float32" cpuBLASThreading:nil
      layers:layers kvHeads:heads headDim:dim capacity:capacity vocabulary:vocabulary effectiveWeightsID:weights error:error];
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens dataPath:(nullable NSString *)dataPath cacheWindow:(NSInteger)cacheWindow
    decodeOnCPU:(BOOL)decodeOnCPU kvDtype:(NSString *)kvDtype layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:verifyPath verifyTokens:verifyTokens
      dataPath:dataPath cacheWindow:cacheWindow decodeOnCPU:decodeOnCPU kvDtype:kvDtype cpuBLASThreading:nil
      layers:layers kvHeads:heads headDim:dim capacity:capacity vocabulary:vocabulary effectiveWeightsID:weights error:error];
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens dataPath:(nullable NSString *)dataPath cacheWindow:(NSInteger)cacheWindow
    decodeOnCPU:(BOOL)decodeOnCPU kvDtype:(NSString *)kvDtype cpuBLASThreading:(nullable NSString *)cpuBLASThreading
    layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:verifyPath verifyTokens:verifyTokens
      dataPath:dataPath cacheWindow:cacheWindow decodeOnCPU:decodeOnCPU kvDtype:kvDtype cpuBLASThreading:cpuBLASThreading
      layers:layers kvHeads:heads headDim:dim capacity:capacity vocabulary:vocabulary effectiveWeightsID:weights prefillTokens:128 error:error];
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens dataPath:(nullable NSString *)dataPath cacheWindow:(NSInteger)cacheWindow
    decodeOnCPU:(BOOL)decodeOnCPU kvDtype:(NSString *)kvDtype cpuBLASThreading:(nullable NSString *)cpuBLASThreading
    layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights prefillTokens:(NSInteger)prefillTokens error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:verifyPath verifyTokens:verifyTokens
      dataPath:dataPath cacheWindow:cacheWindow decodeOnCPU:decodeOnCPU kvDtype:kvDtype cpuBLASThreading:cpuBLASThreading
      layers:layers kvHeads:heads headDim:dim capacity:capacity vocabulary:vocabulary effectiveWeightsID:weights
      prefillTokens:prefillTokens allocationLimit:0 kvPayloadBudgetBytes:nil error:error];
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens dataPath:(nullable NSString *)dataPath cacheWindow:(NSInteger)cacheWindow
    decodeOnCPU:(BOOL)decodeOnCPU kvDtype:(NSString *)kvDtype cpuBLASThreading:(nullable NSString *)cpuBLASThreading
    layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights prefillTokens:(NSInteger)prefillTokens
    allocationLimit:(NSInteger)allocationLimit kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes
    prefillMaxQueuedChunksOverride:(nullable NSNumber *)queuedOverride
    nativeExecutionPermit:(nullable TFETNativeExecutionPermit *)permit error:(NSError **)error {
  self = [super init]; if (!self) return nil;
  BOOL loading = NO;
  try {
    need(!permit || [permit isKindOfClass:TFETNativeExecutionPermit.class], "Invalid native execution permission");
    _nativeExecutionPermit = permit;
    [self checkNativeExecutionPermit];
    need(!permit || (!decodeOnCPU && !cacheWindow && !verifyPath && verifyTokens == 0),
         "Native execution permission requires a full-history GPU target without verifier");
    // The legacy path alone consults the environment. An explicit queue limit
    // takes precedence even when the process environment contains invalid data.
    const auto queued = queuedOverride ? exactInteger(queuedOverride) : prefillMaxQueuedChunks();
    need(queued >= 1 && queued <= 64, "Prefill maximum queued chunks must be an integer1..64");
    _prefillMaxQueuedChunks = int(queued);
    for (NSInteger x : {layers, heads, dim, capacity, vocabulary}) need(x > 0 && x <= INT32_MAX, "Invalid resident geometry");
    need(capacity >= 128 && weights.length && prefillPath.length && decodePath.length, "Invalid resident package fields");
    need(prefillTokens == 128 || prefillTokens == 512, "Resident prefill must be fixedT128 or fixedT512");
    need(prefillTokens <= capacity && (prefillTokens == 128 ||
         (!cacheWindow && !verifyPath && verifyTokens == 0)),
         "T512 requires full-history target without verifier and sufficient compiled capacity");
    need(dataPath == nil || ([dataPath isKindOfClass:NSString.class] && dataPath.length), "External weights path must be nonempty when provided");
    need((verifyPath == nil && verifyTokens == 0) ||
         ([verifyPath isKindOfClass:NSString.class] && verifyPath.length && verifyTokens > 0 && verifyTokens <= capacity),
         "Verifier path and fixed token count must be provided together");
    need(cacheWindow == 0 || (cacheWindow >= 128 && cacheWindow <= capacity && cacheWindow <= INT32_MAX - 127), "Invalid explicit draft cache window");
    need(cacheWindow == 0 || verifyTokens <= 128, "Windowed session permits at most128-token blocks");
    need(allocationLimit >= 0 && allocationLimit <= capacity, "Request allocation limit outside compiled capacity");
    need(allocationLimit == 0 || (!cacheWindow && !decodeOnCPU && !verifyPath && verifyTokens == 0),
         "Request-bound allocation requires full-history GPU target without verifier");
    need(!kvPayloadBudgetBytes || allocationLimit > 0, "KV payload budget requires request-bound allocation");
    need([kvDtype isEqualToString:@"float32"] || [kvDtype isEqualToString:@"float16"], "Cache precision must be float32 or float16");
    need(cpuBLASThreading == nil || (decodeOnCPU && ([cpuBLASThreading isEqualToString:@"default"] ||
         [cpuBLASThreading isEqualToString:@"single"] || [cpuBLASThreading isEqualToString:@"multi"])), "Invalid CPU-only BLAS threading policy");
    need(!decodeOnCPU || mlx::core::is_available(mlx::core::Device(mlx::core::Device::cpu)), "MLX CPU kernels are unavailable; rebuild runtime with CPU enabled");
    _layers = int(layers); _heads = int(heads); _dim = int(dim); _capacity = int(capacity); _vocabulary = int(vocabulary);
    _verifyTokens = int(verifyTokens); _decodeOnCPU = decodeOnCPU; _prefillTokens = int(prefillTokens);
    _kvDtype = [kvDtype copy]; _kvScalar = int([kvDtype isEqualToString:@"float16"] ? ScalarType::Half : ScalarType::Float);
    _kvElementBytes = _kvScalar == int(ScalarType::Half) ? 2 : 4;
    _cpuBLASThreading = decodeOnCPU ? [(cpuBLASThreading ?: @"default") copy] : @"not_applicable";
    _requestBounded = allocationLimit > 0; _kvPayloadBudgetBytes = [kvPayloadBudgetBytes copy];
    _cacheWindow = int(cacheWindow); _physicalCapacity = _cacheWindow ? _cacheWindow + 127 : (_requestBounded ? int(allocationLimit) : _capacity);
    need(size_t(std::max(1, _verifyTokens)) <= SIZE_MAX / sizeof(float) / size_t(_vocabulary), "Verifier logits size overflow");
    _weights = [weights copy]; _sessionID = NSUUID.UUID.UUIDString; _elements = 1;
    for (int x : {_layers, _heads, _physicalCapacity, _dim}) {
      need(_elements <= SIZE_MAX / size_t(x), "Resident state size overflow"); _elements *= size_t(x);
    }
    need(_elements <= SIZE_MAX / (2 * size_t(_kvElementBytes)), "Resident paired-state byte size overflow");
    if (_kvPayloadBudgetBytes) {
      const auto budget = exactInteger(_kvPayloadBudgetBytes);
      need(budget > 0 && uint64_t(budget) >= _elements * size_t(_kvElementBytes) * 2,
           "KV payload budget cannot hold complete requested history");
    }
    cache::CacheConfig config{}; config.n_layers = _layers; config.capacity = _capacity;
    if (_requestBounded) config.allocation_capacity = _physicalCapacity;
    config.initial_capacity = std::min(512, _capacity); config.max_write = std::max(_prefillTokens, _verifyTokens); config.kv_dtype = _kvScalar;
    cache::LayerPolicy policy{};
    if (_cacheWindow) { policy.kind = cache::LayerPolicy::Kind::Ring; policy.window = _cacheWindow; }
    config.layers.push_back({policy, _heads, _dim}); need(cache::valid(config), "Invalid resident cache configuration");
    [self checkNativeExecutionPermit];
    runtime_init(); _core = std::make_unique<Core>(config, prefillPath.fileSystemRepresentation, decodePath.fileSystemRepresentation,
                                                verifyPath ? verifyPath.fileSystemRepresentation : nullptr,
                                                dataPath ? dataPath.fileSystemRepresentation : nullptr, bool(_decodeOnCPU),
                                                cpuBLASThreading ? cpuBLASThreading.UTF8String : nullptr);
    [self checkNativeExecutionPermit];
    loading = YES;
    validateModule(_core->prefill, *_core, _prefillTokens, 1, _layers, _heads, _dim, _capacity, _vocabulary, _kvScalar, _prefillTokens, _nativeExecutionPermit);
    diagnosticMemory("prefill_module_loaded");
    [self checkNativeExecutionPermit];
    validateModule(_core->decode, *_core, 1, 1, _layers, _heads, _dim, _capacity, _vocabulary, _kvScalar, _prefillTokens, _nativeExecutionPermit);
    diagnosticMemory("decode_module_loaded");
    if (_core->verify) {
      [self checkNativeExecutionPermit];
      validateModule(*_core->verify, *_core, _verifyTokens, _verifyTokens, _layers, _heads, _dim, _capacity, _vocabulary, _kvScalar, _prefillTokens, _nativeExecutionPermit);
      diagnosticMemory("verify_module_loaded");
    }
    return self;
  } catch (const NativeExecutionCancelled&) {
    [self finishNativeCancellation:error dispatched:loading]; return nil;
  } catch (const std::exception& e) {
    if (loading) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:loading]; return nil;
  } catch (...) {
    if (loading) [self failDispatched];
    [self setError:error message:@"Resident initialization failed" dispatched:loading]; return nil;
  }
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens dataPath:(nullable NSString *)dataPath cacheWindow:(NSInteger)cacheWindow
    decodeOnCPU:(BOOL)decodeOnCPU kvDtype:(NSString *)kvDtype cpuBLASThreading:(nullable NSString *)cpuBLASThreading
    layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights prefillTokens:(NSInteger)prefillTokens
    allocationLimit:(NSInteger)allocationLimit kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:verifyPath verifyTokens:verifyTokens
      dataPath:dataPath cacheWindow:cacheWindow decodeOnCPU:decodeOnCPU kvDtype:kvDtype cpuBLASThreading:cpuBLASThreading
      layers:layers kvHeads:heads headDim:dim capacity:capacity vocabulary:vocabulary effectiveWeightsID:weights
      prefillTokens:prefillTokens allocationLimit:allocationLimit kvPayloadBudgetBytes:kvPayloadBudgetBytes
      prefillMaxQueuedChunksOverride:nil nativeExecutionPermit:nil error:error];
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens dataPath:(nullable NSString *)dataPath cacheWindow:(NSInteger)cacheWindow
    decodeOnCPU:(BOOL)decodeOnCPU kvDtype:(NSString *)kvDtype cpuBLASThreading:(nullable NSString *)cpuBLASThreading
    layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights prefillTokens:(NSInteger)prefillTokens
    allocationLimit:(NSInteger)allocationLimit kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes
    prefillMaxQueuedChunks:(NSInteger)maxQueuedChunks error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:verifyPath verifyTokens:verifyTokens
      dataPath:dataPath cacheWindow:cacheWindow decodeOnCPU:decodeOnCPU kvDtype:kvDtype cpuBLASThreading:cpuBLASThreading
      layers:layers kvHeads:heads headDim:dim capacity:capacity vocabulary:vocabulary effectiveWeightsID:weights
      prefillTokens:prefillTokens allocationLimit:allocationLimit kvPayloadBudgetBytes:kvPayloadBudgetBytes
      prefillMaxQueuedChunksOverride:@(maxQueuedChunks) nativeExecutionPermit:nil error:error];
}
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath decodePath:(NSString *)decodePath
    verifyPath:(nullable NSString *)verifyPath verifyTokens:(NSInteger)verifyTokens dataPath:(nullable NSString *)dataPath cacheWindow:(NSInteger)cacheWindow
    decodeOnCPU:(BOOL)decodeOnCPU kvDtype:(NSString *)kvDtype cpuBLASThreading:(nullable NSString *)cpuBLASThreading
    layers:(NSInteger)layers kvHeads:(NSInteger)heads headDim:(NSInteger)dim capacity:(NSInteger)capacity
    vocabulary:(NSInteger)vocabulary effectiveWeightsID:(NSString *)weights prefillTokens:(NSInteger)prefillTokens
    allocationLimit:(NSInteger)allocationLimit kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes
    prefillMaxQueuedChunks:(NSInteger)maxQueuedChunks
    nativeExecutionPermit:(nullable TFETNativeExecutionPermit *)permit error:(NSError **)error {
  return [self initWithPrefillPath:prefillPath decodePath:decodePath verifyPath:verifyPath verifyTokens:verifyTokens
      dataPath:dataPath cacheWindow:cacheWindow decodeOnCPU:decodeOnCPU kvDtype:kvDtype cpuBLASThreading:cpuBLASThreading
      layers:layers kvHeads:heads headDim:dim capacity:capacity vocabulary:vocabulary effectiveWeightsID:weights
      prefillTokens:prefillTokens allocationLimit:allocationLimit kvPayloadBudgetBytes:kvPayloadBudgetBytes
      prefillMaxQueuedChunksOverride:@(maxQueuedChunks) nativeExecutionPermit:permit error:error];
}
- (BOOL)configureNativeExecutionPermit:(TFETNativeExecutionPermit *)permit
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex);
  try {
    [self admit:generation];
    need([permit isKindOfClass:TFETNativeExecutionPermit.class],
         "New native execution permission must be present");
    need(!_decodeOnCPU && !_cacheWindow && !_core->verify && !_verifyTokens &&
         !_features && !_selectionChannel && !_selectionUsed && !_retainedCompacted &&
         !_core->active && _core->prefillBatch.empty() && !_core->importing &&
         _core->storage->length() == _position,
         "Native execution permission requires an idle full-history GPU target");
    if (permit.isCancelled) {
      // This concerns only the rejected replacement. Do not run the owner's
      // cancellation cleanup or overwrite its existing permission/metadata.
      if (error) *error = [NSError errorWithDomain:@"TriFuse.Resident" code:3 userInfo:@{
        NSLocalizedDescriptionKey:@"Replacement native execution permission cancelled",
        @"cancelled":@YES, @"nativeExecutionCancelled":@YES,
        @"completion":@"not_dispatched", @"cancellationDrained":@NO,
        @"sessionInvalid":@(_invalid), @"committedPosition":@(_position), @"generation":@(_generation)}];
      return NO;
    }
    _nativeExecutionPermit = permit;
    _nativeCancellationCompletion = nil;
    _nativeCancellationKnownIdle = NO;
    return YES;
  } catch (const std::exception& e) {
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:NO]; return NO;
  } catch (...) {
    [self setError:error message:@"Native execution permission binding failed" dispatched:NO]; return NO;
  }
}
- (BOOL)configurePrefillMaxQueuedChunks:(NSInteger)maxQueuedChunks
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex);
  try {
    [self admit:generation];
    need(maxQueuedChunks >= 1 && maxQueuedChunks <= 64, "Prefill maximum queued chunks must be an integer1..64");
    need(!_core->active && _core->prefillBatch.empty() && !_core->importing &&
         _core->storage->length() == _position,
         "Prefill queue configuration requires a ready owner without pending work");
    // CPU policy publication only: do not reset/drain the cache or invalidate
    // the native prefill proof, and do not consume a KV generation number.
    _prefillMaxQueuedChunks = int(maxQueuedChunks);
    return YES;
  } catch (const std::exception& e) {
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:NO]; return NO;
  } catch (...) {
    [self setError:error message:@"Prefill queue configuration failed" dispatched:NO]; return NO;
  }
}
- (nullable NSData *)forwardTokenIDs:(NSArray<NSNumber *> *)tokens positions:(NSArray<NSNumber *> *)positions
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  return [self forwardTokenIDs:tokens positions:positions nativeConfiguration:nil expectedGeneration:generation error:error];
}
- (nullable NSData *)forwardTokenIDs:(NSArray<NSNumber *> *)tokens positions:(NSArray<NSNumber *> *)positions
    nativeConfiguration:(nullable NSDictionary<NSString *, id> *)configuration
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  return [self executeTokenIDs:tokens positions:positions nativeConfiguration:configuration expectedGeneration:generation verification:NO error:error];
}
- (nullable NSData *)verifyTokenIDs:(NSArray<NSNumber *> *)tokens positions:(NSArray<NSNumber *> *)positions
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  return [self executeTokenIDs:tokens positions:positions nativeConfiguration:nil expectedGeneration:generation verification:YES error:error];
}
- (nullable NSNumber *)forwardGreedyTokenID:(NSNumber *)tokenID position:(NSNumber *)position
    nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  NSNumber *nextToken = nil;
  NSData *completed = [self executeTokenIDs:tokenID ? @[tokenID] : @[]
      positions:position ? @[position] : @[] nativeConfiguration:configuration
      expectedGeneration:generation verification:NO attentionLabelLayers:nil
      attentionLabelOutput:nullptr greedyOutput:&nextToken error:error];
  return completed ? nextToken : nil;
}
- (void)updateGreedyPipelineReceipt:(NSString *)state {
  if (!_core || !_core->greedyPipeline || !_greedyPipelineReceipt) return;
  const auto& pipeline = *_core->greedyPipeline;
  _greedyPipelineReceipt[@"state"] = state;
  _greedyPipelineReceipt[@"submittedTokens"] = @(pipeline.submitted);
  _greedyPipelineReceipt[@"returnedTokens"] = @(pipeline.returned);
  _greedyPipelineReceipt[@"completedTokens"] = @(pipeline.completed);
  _greedyPipelineReceipt[@"decodeAllocatorClears"] = @(pipeline.decode_clears);
  _greedyPipelineReceipt[@"committedPosition"] = @(_position);
  _greedyPipelineReceipt[@"generation"] = @(_generation);
}
- (NSDictionary<NSString *, id> *)greedyPipelineReceipt {
  std::lock_guard<std::mutex> lock(_mutex);
  return _greedyPipelineReceipt ? [_greedyPipelineReceipt copy] : @{@"state":@"inactive"};
}
- (BOOL)beginGreedyPipelineWithFirstToken:(NSNumber *)firstToken
    nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex);
  try {
    [self admit:generation]; [self checkNativeExecutionPermit];
    const auto first = exactInteger(firstToken);
    const auto config = parseNativeConfiguration(configuration);
    need(backend::is_native_moe_policy_architecture(config.architecture) &&
         config.layers == _layers && config.kv_heads == _heads && config.head_dim == _dim &&
         config.vocabulary == _vocabulary && config.capacity == _capacity &&
         first >= 0 && first < _vocabulary && _position > 0 && _position < _physicalCapacity &&
         !_core->active && _core->prefillBatch.empty() && !_core->importing &&
         !_features && !_selectionChannel && !_decodeOnCPU && !_cacheWindow && !_retainedCompacted &&
         !_core->verify && _verifyTokens == 0 && _core->storage->has_complete_history_positions() &&
         _core->storage->length() == _position && _core->nativePrefillConfiguration &&
         sameNativeConfiguration(config, *_core->nativePrefillConfiguration),
         "Greedy pipeline requires a completed matching native Granite prefill owner");
    const auto meta = _core->decode.method_meta("forward");
    need(meta.ok() && meta->num_instructions() == 1 && meta->num_backends() == 1 &&
         meta->uses_backend(backend::kMLXBackendId), "Greedy pipeline requires pure MLX T1");
    auto pipeline = std::make_unique<GreedyPipeline>();
    pipeline->config = config; pipeline->generation = generation;
    pipeline->prompt_position = _position; pipeline->first_token = int(first);
    _greedyPipelineReceipt = [@{@"schema":@"native_greedy_pipeline_v1", @"state":@"primed",
      @"promptPosition":@(_position), @"entryGeneration":@(generation), @"generation":@(generation),
      @"submittedTokens":@0, @"returnedTokens":@0, @"completedTokens":@0,
      @"submitBeforeReadCount":@0, @"committedPosition":@(_position),
      @"initialPrediction":@"completed_prefill_cpu_argmax", @"decodeClearInterval":@256,
      @"cancellationCompletion":NSNull.null,
      @"scope":@"GPU checked-token dependency; consuming T1 submitted before prior token read. Provisional cache is not a committed endpoint; finish drains without another model call. Prefill still uses existing native final-block path."} mutableCopy];
    _core->greedyPipeline = std::move(pipeline);
    return YES;
  } catch (const NativeExecutionCancelled&) {
    [self finishNativeCancellation:error dispatched:NO]; return NO;
  } catch (const std::exception& e) {
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:NO]; return NO;
  } catch (...) {
    [self setError:error message:@"Greedy pipeline admission failed" dispatched:NO]; return NO;
  }
}
- (nullable NSDictionary<NSString *, id> *)beginPythonPolicyTokenIDs:(NSArray<NSNumber *> *)tokens
    nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation]; [self checkNativeExecutionPermit];
    const auto config = parseNativeConfiguration(configuration);
    need(backend::is_native_moe_policy_architecture(config.architecture) &&
         config.layers == _layers && config.kv_heads == _heads && config.head_dim == _dim &&
         config.vocabulary == _vocabulary && config.capacity == _capacity &&
         [tokens isKindOfClass:NSArray.class] && tokens.count > 0 &&
         tokens.count < size_t(_physicalCapacity) && _position == 0 && _prefillTokens == 512 &&
         !_core->active && _core->prefillBatch.empty() && !_core->importing &&
         !_features && !_selectionChannel && !_decodeOnCPU && !_cacheWindow && !_retainedCompacted &&
         !_core->verify && _verifyTokens == 0 && _core->storage->has_full_history_storage() &&
         _core->storage->length() == 0,
         "Python policy requires a fresh full-history Granite T512/T1 owner");
    tokens = [tokens copy];
    for (NSNumber *value in tokens) {
      const auto id = exactInteger(value); need(id >= 0 && id < _vocabulary, "Prompt token outside vocabulary");
    }
    const int prompt = int(tokens.count);
    if (prompt > 1) {
      NSData *prefix = [self prefillTokenIDsLocked:[tokens subarrayWithRange:NSMakeRange(0, prompt - 1)]
          nativeConfiguration:configuration policyCacheOnly:YES expectedGeneration:generation error:error];
      if (!prefix) return nil; // Preserve the helper's original error/cleanup.
    } else {
      _core->ensurePolicyContext();
    }
    [self checkNativeExecutionPermit];
    const auto meta = _core->decode.method_meta("forward");
    need(meta.ok() && meta->num_instructions() == 1 && meta->num_backends() == 1 &&
         meta->uses_backend(backend::kMLXBackendId), "Python policy requires pure native T1");
    auto pipeline = std::make_unique<GreedyPipeline>();
    pipeline->config = config; pipeline->generation = _generation;
    pipeline->prompt_position = prompt; pipeline->pending_prompt_tokens = 1;
    pipeline->policy = _core->policyContext;
    _core->greedyPipeline = std::move(pipeline);
    auto& state = *_core->greedyPipeline;
    _greedyPipelineReceipt = [@{@"schema":@"native_python_policy_v2", @"state":@"priming",
      @"promptPosition":@(prompt), @"entryGeneration":@(_generation), @"generation":@(_generation),
      @"submittedTokens":@0, @"returnedTokens":@0, @"completedTokens":@0,
      @"submitBeforeReadCount":@0, @"committedPosition":@(_position),
      @"initialPrediction":@"prompt_last_T1_gpu_checked_token", @"promptLastT1Calls":@1,
      @"prefillCacheOnlyTokens":@(prompt - 1), @"prefillCacheOnlyChunks":@((prompt - 1 + 511) / 512),
      @"prefillQueueDepth":@1, @"prefillClearInterval":@1, @"decodeClearInterval":@256,
      @"singlePolicyStream":@YES, @"cpuArgmaxCalls":@0, @"cancellationCompletion":NSNull.null,
      @"ownerPolicyContextCreations":@(_core->policyContextCreations),
      @"scope":@"C++ translation of Python generate_tokens: prompt[:-1] cache-only T512 queue1/drain/clear1; prompt-last T1 and first prediction stay on GPU; checked-token submit-before-yield decode; exact normal/early-close drain ordering."} mutableCopy];
    auto call = std::make_shared<Invocation>();
    call->tokens.push_back(exactInteger(tokens.lastObject)); call->positions.push_back(prompt - 1);
    call->tensors.push_back(make_tensor_ptr({1, 1}, call->tokens.data(), ScalarType::Long));
    call->tensors.push_back(make_tensor_ptr({1}, call->positions.data(), ScalarType::Long));
    for (const auto& tensor : call->tensors) call->values.emplace_back(*tensor);
    state.active = call;
    backend::ResidentMoEPolicyScope policyScope(state.policy);
    backend::NativeLlamaPrefillScope nativeScope(_core->storage.get(), config);
    backend::ResidentGreedySubmissionScope submission(_core->storage.get(), _vocabulary,
        prompt - 1, nullptr, call, true);
    state.submitted_prediction = submission.cleanup_ticket();
    [self checkNativeExecutionPermit]; dispatched = YES; state.prompt_submitted = true;
    const auto result = _core->decode.forward(call->values);
    need(result.ok(), "Python policy initial prediction submission failed");
    state.prediction = submission.submitted_ticket();
    state.submitted_prediction.reset(); state.active.reset();
    need(_core->storage->length() == prompt, "Python policy prompt KV append differs");
    [self updateGreedyPipelineReceipt:@"primed"];
    return [_greedyPipelineReceipt copy];
  } catch (const NativeExecutionCancelled&) {
    [self finishNativeCancellation:error dispatched:dispatched]; return nil;
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return nil;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Python policy prompt preparation failed" dispatched:dispatched]; return nil;
  }
}
- (nullable NSData *)prefillPythonPolicyTokenIDs:(NSArray<NSNumber *> *)tokens
    nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  // The quality API returns the final prompt row. It uses the same prefix
  // traversal, but does not start generation or sample a token.
  std::unique_lock<std::mutex> lock(_mutex);
  try {
    [self admit:generation]; [self checkNativeExecutionPermit];
    need([tokens isKindOfClass:NSArray.class] && tokens.count > 1 && tokens.count <= size_t(_physicalCapacity),
         "Policy gold prefill requires at least two prompt tokens within capacity");
    tokens = [tokens copy];
    const auto last = exactInteger(tokens.lastObject);
    need(last >= 0 && last < _vocabulary, "Last prompt token outside vocabulary");
    if (![self prefillTokenIDsLocked:[tokens subarrayWithRange:NSMakeRange(0, tokens.count - 1)]
         nativeConfiguration:configuration policyCacheOnly:YES expectedGeneration:generation error:error]) return nil;
    const uint64_t tailGeneration = _generation;
    const int position = _position;
    auto context = _core->policyContext;
    lock.unlock();
    backend::ResidentMoEPolicyScope policyScope(context);
    // This method independently rechecks generation/owner under its mutex;
    // another owner mutation between stages cannot silently change the prompt.
    return [self forwardTokenIDs:@[@(last)] positions:@[@(position)] nativeConfiguration:configuration
        expectedGeneration:tailGeneration error:error];
  } catch (const NativeExecutionCancelled&) {
    if (!lock.owns_lock()) lock.lock();
    [self finishNativeCancellation:error dispatched:NO]; return nil;
  } catch (const std::exception& e) {
    if (!lock.owns_lock()) lock.lock();
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:NO]; return nil;
  } catch (...) {
    if (!lock.owns_lock()) lock.lock();
    [self setError:error message:@"Python policy gold prefill failed" dispatched:NO]; return nil;
  }
}
- (nullable NSNumber *)nextGreedyPipelineExpectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    need(!_invalid && _core && _core->greedyPipeline, "No active greedy pipeline");
    auto& pipeline = *_core->greedyPipeline;
    need(generation == _generation && generation == pipeline.generation, "Stale greedy pipeline generation");
    dispatched = pipeline.submitted > 0 || pipeline.prompt_submitted;
    [self checkNativeExecutionPermit];
    need(pipeline.submitted == pipeline.returned && !pipeline.active && !pipeline.submitted_prediction &&
         pipeline.prompt_position + pipeline.submitted < _physicalCapacity &&
         uint64_t(pipeline.returned + pipeline.pending_prompt_tokens) < uint64_t(INT64_MAX) - _generation &&
         _core->storage->length() == pipeline.prompt_position + pipeline.submitted,
         "Greedy pipeline pending delivery or exhausted/mismatched cache");
#ifdef TRIFUSE_ENABLE_COLD_KV
    if (_core->coldInstaller && _core->coldInstaller->encoder) {
      // The owner is locked. No scope/forward has consumed these roots yet.
      // Ordinary checks/feed are nonblocking; bounded progress and publication
      // explicitly drain the latest ticket and record the resulting wait.
      dispatched = YES;
      _core->coldInstaller->tick(*_core->storage, [&] {
        if (pipeline.prediction) (void)pipeline.prediction->wait_token();
        _core->storage->synchronize_storage();
        if (pipeline.policy) pipeline.policy->synchronize();
      });
    }
#endif
    if (pipeline.clear_due) {
      backend::clear_greedy_allocator_cache(); pipeline.clear_due = false; ++pipeline.decode_clears;
    }
    auto call = std::make_shared<Invocation>();
    // A valid host placeholder preserves the fixed ET ABI. With a prediction
    // ticket only its GPU safe scalar reaches the native embedding operation.
    call->tokens.push_back(pipeline.prediction ? 0 : pipeline.first_token);
    call->positions.push_back(pipeline.prompt_position + pipeline.submitted);
    call->tensors.push_back(make_tensor_ptr({1, 1}, call->tokens.data(), ScalarType::Long));
    call->tensors.push_back(make_tensor_ptr({1}, call->positions.data(), ScalarType::Long));
    for (const auto& tensor : call->tensors) call->values.emplace_back(*tensor);
    pipeline.active = call;
    std::unique_ptr<backend::ResidentMoEPolicyScope> policyScope;
    if (pipeline.policy) policyScope = std::make_unique<backend::ResidentMoEPolicyScope>(pipeline.policy);
    backend::NativeLlamaPrefillScope nativeScope(_core->storage.get(), pipeline.config);
    backend::ResidentGreedySubmissionScope submission(_core->storage.get(), _vocabulary,
        int(call->positions.front()), pipeline.prediction, call);
    // Store the unsealed ticket before forward: a partial submission failure
    // must keep its roots and input lease in the quarantined Core too.
    pipeline.submitted_prediction = submission.cleanup_ticket();
    [self checkNativeExecutionPermit]; dispatched = YES;
    const auto result = _core->decode.forward(call->values);
    need(result.ok(), "Greedy pipeline T1 submission failed");
    pipeline.submitted_prediction = submission.submitted_ticket();
    ++pipeline.submitted;
    _greedyPipelineReceipt[@"submitBeforeReadCount"] = @(pipeline.submitted);
    [self updateGreedyPipelineReceipt:@"awaiting_previous_scalar"];
    const int64_t token = pipeline.prediction ? pipeline.prediction->wait_token() : pipeline.first_token;
    // Reading the prior prediction completes the previous consuming T1, not
    // the T1 just submitted for this token. Keep that distinction observable.
    pipeline.completed = pipeline.submitted - 1;
    need(token >= 0 && token < _vocabulary, "Greedy pipeline token outside vocabulary");
    pipeline.prediction = std::move(pipeline.submitted_prediction);
    pipeline.active.reset();
    pipeline.clear_due = pipeline.returned % 256 == 0;
    ++pipeline.returned;
    [self updateGreedyPipelineReceipt:@"active"];
    return @(token);
  } catch (const NativeExecutionCancelled&) {
    [self updateGreedyPipelineReceipt:@"cancelling"];
    [self finishNativeCancellation:error dispatched:dispatched]; return nil;
  } catch (const std::exception& e) {
    [self updateGreedyPipelineReceipt:@"failed"];
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return nil;
  } catch (...) {
    [self updateGreedyPipelineReceipt:@"failed"];
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Greedy pipeline next failed" dispatched:dispatched]; return nil;
  }
}
- (nullable NSDictionary<NSString *, id> *)finishGreedyPipelineExpectedGeneration:(uint64_t)generation
    error:(NSError **)error {
  return [self finishGreedyPipelineExpectedGeneration:generation resumeLastYield:NO error:error];
}
- (nullable NSDictionary<NSString *, id> *)finishGreedyPipelineExpectedGeneration:(uint64_t)generation
    resumeLastYield:(BOOL)resumeLastYield error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    if (!_invalid && _core && !_core->greedyPipeline &&
        [_greedyPipelineReceipt[@"state"] isEqual:@"finished"] &&
        exactInteger(_greedyPipelineReceipt[@"entryGeneration"]) == int64_t(generation) &&
        exactInteger(_greedyPipelineReceipt[@"generation"]) == int64_t(_generation))
      return [_greedyPipelineReceipt copy];
    need(!_invalid && _core && _core->greedyPipeline, "No active greedy pipeline to finish");
    auto& pipeline = *_core->greedyPipeline;
    need(generation == _generation && generation == pipeline.generation, "Stale greedy pipeline finish");
    dispatched = pipeline.submitted > 0 || pipeline.prompt_submitted;
    [self checkNativeExecutionPermit];
    need(pipeline.submitted == pipeline.returned && !pipeline.active && !pipeline.submitted_prediction,
         "Greedy pipeline has an unsuccessful pending delivery");
    [self updateGreedyPipelineReceipt:@"draining"];
    // Python's code after yield runs on normal exhaustion, not generator.close.
    if (resumeLastYield && pipeline.clear_due) {
      backend::clear_greedy_allocator_cache(); ++pipeline.decode_clears;
    }
    // The last consuming T1 has an unscored prediction. Validate it on normal
    // close/EOS as well; cancellation uses submitted-only drainage elsewhere.
    if (pipeline.prediction) (void)pipeline.prediction->wait_token();
    if (pipeline.submitted || pipeline.prompt_submitted) _core->storage->synchronize_storage();
    if (pipeline.policy) pipeline.policy->synchronize();
    need(_core->storage->length() == pipeline.prompt_position + pipeline.returned,
         "Greedy pipeline final KV length differs");
#ifdef TRIFUSE_ENABLE_COLD_KV
    if (_core->coldInstaller) _core->coldInstaller->tick(*_core->storage, [&] {
      _core->storage->synchronize_storage();
      if (pipeline.policy) pipeline.policy->synchronize();
    }, false); // Close never initiates or forces unfinished encoding.
#endif
    pipeline.completed = pipeline.returned;
    _position = pipeline.prompt_position + pipeline.returned;
    _generation += uint64_t(pipeline.returned + pipeline.pending_prompt_tokens);
    _greedyPipelineReceipt[@"resumeLastYield"] = @(resumeLastYield);
    [self updateGreedyPipelineReceipt:@"finished"];
    _core->greedyPipeline.reset();
#ifdef TRIFUSE_ENABLE_COLD_KV
    const auto finalKV = _core->storage->storage_stats();
    _greedyPipelineReceipt[@"finalKVStorage"] = @{@"payloadBytes":@(finalKV.payload_bytes),
      @"bufferBytes":@(finalKV.buffer_bytes), @"coldPayloadBytes":@(finalKV.cold_payload_bytes),
      @"hotPayloadBytes":@(finalKV.hot_payload_bytes), @"originalFP16OwnerBytes":@(finalKV.original_fp16_owner_bytes),
      @"mlxActiveBytes":@(mlx::core::get_active_memory()),
      @"scope":@"completed cache-owned roots, not process peak or allocator return"};
#endif
    return [_greedyPipelineReceipt copy];
  } catch (const NativeExecutionCancelled&) {
    [self updateGreedyPipelineReceipt:@"cancelling"];
    [self finishNativeCancellation:error dispatched:dispatched]; return nil;
  } catch (const std::exception& e) {
    [self updateGreedyPipelineReceipt:@"failed"];
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return nil;
  } catch (...) {
    [self updateGreedyPipelineReceipt:@"failed"];
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Greedy pipeline finish failed" dispatched:dispatched]; return nil;
  }
}
- (nullable NSDictionary<NSString *, id> *)forwardWithAttentionLabelTokenIDs:(NSArray<NSNumber *> *)tokens
    positions:(NSArray<NSNumber *> *)positions nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
    layers:(NSArray<NSNumber *> *)layers expectedGeneration:(uint64_t)generation error:(NSError **)error {
  NSDictionary<NSString *, id> *labels = nil;
  NSData *logits = [self executeTokenIDs:tokens positions:positions nativeConfiguration:configuration
      expectedGeneration:generation verification:NO attentionLabelLayers:layers attentionLabelOutput:&labels
      greedyOutput:nullptr error:error];
  return logits && labels ? @{ @"logits":logits, @"attentionLabels":labels } : nil;
}
- (nullable NSData *)executeTokenIDs:(NSArray<NSNumber *> *)tokens positions:(NSArray<NSNumber *> *)positions
    nativeConfiguration:(nullable NSDictionary<NSString *, id> *)configuration
    expectedGeneration:(uint64_t)generation verification:(BOOL)verification error:(NSError **)error {
  return [self executeTokenIDs:tokens positions:positions nativeConfiguration:configuration
      expectedGeneration:generation verification:verification attentionLabelLayers:nil attentionLabelOutput:nullptr
      greedyOutput:nullptr error:error];
}
- (nullable NSData *)executeTokenIDs:(NSArray<NSNumber *> *)tokens positions:(NSArray<NSNumber *> *)positions
    nativeConfiguration:(nullable NSDictionary<NSString *, id> *)configuration
    expectedGeneration:(uint64_t)generation verification:(BOOL)verification
    attentionLabelLayers:(nullable NSArray<NSNumber *> *)labelLayers
    attentionLabelOutput:(NSDictionary<NSString *, id> * __autoreleasing *)labelOutput
    greedyOutput:(NSNumber * __autoreleasing *)greedyOutput error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation];
    [self checkNativeExecutionPermit];
    need([tokens isKindOfClass:NSArray.class] && [positions isKindOfClass:NSArray.class], "Token/position arrays required");
    need(!_nativeExecutionPermit || (configuration && !verification && !labelLayers && tokens.count == 1),
         "Permission-bound forward requires native target T1 without labels or verification");
    tokens = [tokens copy]; positions = [positions copy];
    const size_t count = tokens.count;
    if (verification) {
      need(_core->verify && _verifyTokens > 0, "Resident verifier is not configured");
      need(count == size_t(_verifyTokens) && positions.count == count, "Verifier requires exactly Tk token/position inputs");
    } else {
      need((count == 1 || count == size_t(_prefillTokens)) && positions.count == count, "Only T1/fixed-prefill token/position inputs are supported");
    }
    need(!_retainedCompacted || (!verification && count == 1 && configuration != nil),
         "Retained diagnostic history requires explicit native T1; exported/prefill/verifier execution is unsupported");
    need(!_selectionChannel || (!verification && count == 1 && configuration != nil),
         "Attention selection requires native T1 or the native batch prefill API");
    need(_position >= 0 && _position <= _capacity && count <= size_t(_capacity - _position), "Resident append exceeds capacity");
    need(!_requestBounded || (_position <= _physicalCapacity && count <= size_t(_physicalCapacity - _position)),
         "Resident append exceeds request allocation limit");
    need(_core->storage->length() == _position, "Resident cache committed length differs");
    std::shared_ptr<backend::AttentionLabelTicket> labelTicket;
    std::unique_ptr<backend::ResidentAttentionLabelScope> labelScope;
    need(bool(labelOutput) == bool(labelLayers), "Incomplete offline label output configuration");
    if (labelLayers) {
      need(labelOutput && configuration && !verification && count == 1 && !_selectionChannel && !_features &&
           !_decodeOnCPU && !_cacheWindow && !_retainedCompacted && _core->storage->has_full_history_storage(),
           "Offline attention labels require dense native GPU T1 without advisers/features");
      need([labelLayers isKindOfClass:NSArray.class] && labelLayers.count > 0, "Attention label layers required");
      backend::ResidentAttentionLabelConfig labels;
      int previous = -1;
      for (id item in [labelLayers copy]) {
        const auto layer = exactInteger(item);
        need(layer > previous && layer < _layers, "Attention label layers must be sorted unique valid indices");
        labels.layers.push_back(int(layer)); previous = int(layer);
      }
      labelScope = std::make_unique<backend::ResidentAttentionLabelScope>(_core->storage.get(), labels,
          [&](std::shared_ptr<backend::AttentionLabelTicket> ticket) {
            need(!labelTicket, "Multiple attention label tickets for one T1"); labelTicket = std::move(ticket);
          });
    }
    std::unique_ptr<backend::NativeLlamaPrefillScope> nativeScope;
    if (configuration) {
      need(!verification && count == 1 && !_decodeOnCPU && !_cacheWindow,
           "Native forward requires one T1 GPU append with flat storage");
      const auto config = parseNativeConfiguration(configuration);
      need(config.architecture == backend::NativeModelArchitecture::StandardLlamaTied ||
           (!_features && !_selectionChannel && !labelLayers && !_retainedCompacted),
           "This architecture requires full-history native execution without advisers or sparse retention");
      need(config.layers == _layers && config.kv_heads == _heads && config.head_dim == _dim && config.vocabulary == _vocabulary && config.capacity == _capacity,
           "Native forward configuration differs from cache owner");
      need(!_retainedCompacted || (_core->nativePrefillConfiguration &&
           sameNativeConfiguration(config, *_core->nativePrefillConfiguration)),
           "Retained native T1 configuration differs from completed prefill");
      const auto meta = _core->decode.method_meta("forward");
      need(meta.ok() && meta->num_instructions() == 1 && meta->num_backends() == 1 && meta->uses_backend(backend::kMLXBackendId),
           "Native forward requires one pure MLX delegate instruction");
      nativeScope = std::make_unique<backend::NativeLlamaPrefillScope>(_core->storage.get(), config);
    }
    std::unique_ptr<backend::ResidentGreedyOutputScope> greedyScope;
    if (greedyOutput) {
      need(nativeScope && count == 1 && !verification && !labelLayers && !_selectionChannel &&
           !_features && !_decodeOnCPU && !_cacheWindow && !_retainedCompacted &&
           _core->storage->has_complete_history_positions(),
           "GPU greedy selection requires full-history native GPU T1 without advisers or labels");
      greedyScope = std::make_unique<backend::ResidentGreedyOutputScope>(_core->storage.get(), _vocabulary);
    }
    auto call = std::make_unique<Invocation>(); call->tokens.reserve(count); call->positions.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const auto t = exactInteger(tokens[i]), p = exactInteger(positions[i]);
      need(t >= 0 && t < _vocabulary, "Token outside vocabulary");
      need(p >= 0 && p <= INT32_MAX && p == int64_t(_position) + int64_t(i), "Positions must append committed history");
      call->tokens.push_back(t); call->positions.push_back(p);
    }
    call->tensors.push_back(make_tensor_ptr({1, int32_t(count)}, call->tokens.data(), ScalarType::Long));
    call->tensors.push_back(make_tensor_ptr({int32_t(count)}, call->positions.data(), ScalarType::Long));
    for (const auto& tensor : call->tensors) call->values.emplace_back(*tensor);
    if (_selectionMailbox) {
      auto selection = _selectionMailbox->take_latest(_position);
      if (selection != _activeSelection) {
        _core->storage->set_attention_selection(selection);
        _activeSelection = std::move(selection);
      }
    }
    [self checkNativeExecutionPermit];
#ifdef TRIFUSE_ENABLE_COLD_KV
    if (_core->coldInstaller && _core->coldInstaller->encoder) {
      need(configuration && !verification && !labelLayers && count == 1,
           "Cold KV threshold path requires native T1 without labels/verifier");
      dispatched = YES; // A partially published candidate needs quarantine.
      _core->coldInstaller->tick(*_core->storage, [&] {
        _core->storage->synchronize_storage();
        if (_core->policyContext) _core->policyContext->synchronize();
      });
    }
#endif
    _core->active = std::move(call); dispatched = YES;
    const uint64_t selectionCallsBefore = _core->storage->attention_selection_layer_calls();
    Module& module = verification ? *_core->verify : (count == size_t(_prefillTokens) ? _core->prefill : _core->decode);
    const auto result = module.forward(_core->active->values);
    need(result.ok() && result->size() == 1 && result->at(0).isTensor(), "Resident forward failed");
    const auto& out = result->at(0).toTensor();
    const int outputTokens = verification ? _verifyTokens : 1;
    const size_t elements = size_t(outputTokens) * size_t(_vocabulary);
    need(out.scalar_type() == ScalarType::Float && out.dim() == 3 && out.size(0) == 1 && out.size(1) == outputTokens && out.size(2) == _vocabulary && out.strides()[2] == 1 && out.strides()[1] == _vocabulary && out.strides()[0] == int64_t(elements) && out.numel() == int64_t(elements) && out.const_data_ptr<float>(), "Resident logits ABI differs");
    backend::ResidentGreedyResult selected;
    if (greedyScope) {
      need(greedyScope->completed(), "GPU greedy output did not complete");
      selected = greedyScope->result();
      need(selected.vocabulary == _vocabulary && selected.token >= 0 && selected.token < _vocabulary,
           "GPU greedy scalar differs from vocabulary");
    }
    // Only the completed backend receipt can establish that the scalar's
    // dependency graph already completed every canonical K/V write. A caller
    // selecting a mode or architecture is not sufficient to skip this barrier.
    if (!greedyScope || !selected.kv_dependencies_completed) {
      _core->storage->synchronize_storage();
    }
    need(_core->storage->length() == _position + int(count), "Resident cache completion length differs");
    NSNumber *nextToken = nil;
    NSData *bytes = nil;
    if (greedyScope) {
      // The private backend scope owns a checked scalar. ET output storage was
      // intentionally not populated, so never read its otherwise valid ABI.
      nextToken = @(selected.token);
      bytes = [NSData data];
    } else {
      const auto *data = out.const_data_ptr<float>();
      for (size_t i = 0; i < elements; ++i) need(std::isfinite(data[i]), "Nonfinite resident logits");
      bytes = [NSData dataWithBytes:data length:elements * sizeof(float)];
    }
    for (size_t i = 0; i < count; ++i) need(_core->active->tokens[i] == exactInteger(tokens[i]) && _core->active->positions[i] == int64_t(_position) + int64_t(i), "Resident inputs mutated");
    need(bytes != nil, "Could not materialize logits");
    if (labelScope) {
      need(bool(labelTicket), "Missing offline attention label ticket");
      const auto batch = labelTicket->wait_and_read();
      need(batch.query_position == _position && batch.full_length == _position + 1,
           "Attention label position differs from consumed token");
      NSMutableArray<NSNumber *> *layers = [NSMutableArray arrayWithCapacity:batch.layers.size()];
      for (int layer : batch.layers) [layers addObject:@(layer)];
      NSData *scores = [NSData dataWithBytes:batch.values.data() length:batch.values.size() * sizeof(float)];
      need(scores != nil, "Could not materialize attention block labels");
      *labelOutput = @{ @"queryPosition":@(batch.query_position), @"fullLength":@(batch.full_length),
          @"coveredLength":@(batch.covered_length), @"blockTokens":@(batch.block_tokens),
          @"blockCount":@(batch.block_count),
          @"queryHeads":@(batch.query_heads), @"kvHeads":@(batch.kv_heads), @"headDim":@(batch.head_dim),
          @"layers":layers, @"scoresFloat32":scores, @"offlineOnly":@YES,
          @"source":[NSString stringWithUTF8String:backend::AttentionLabelBatch::source],
          @"sourcePrecision":[NSString stringWithUTF8String:backend::AttentionLabelBatch::source_precision],
          @"fusedKernelProbabilities":@NO, @"causalDeploymentAdvice":@NO };
    }
    if (_retainedCompacted) _retainedStats = _core->storage->storage_stats();
    if (_selectionChannel) {
      const uint64_t appliedLayers = _core->storage->attention_selection_layer_calls() - selectionCallsBefore;
      need(appliedLayers == 0 || appliedLayers == uint64_t(_layers), "Attention selection applied to incomplete layer set");
      if (appliedLayers) {
        ++_selectionSteps; _selectionUsed = YES;
      } else { ++_selectionFullSteps; }
    }
    // A fully completed admitted operation commits once. A concurrent cancel
    // is observed at the next boundary, never reported as an unconsumed token.
    _position += int(count); ++_generation; _core->active.reset();
    if (nativeScope) {
      // Exact-prefix reuse replays its final prompt token through native T1.
      // Retain the completed adapter proof just as batch prefill does.
      _core->nativePrefillConfiguration = nativeScope->config();
      _core->nativePrefillGeneration = _generation;
    }
    if (greedyOutput) *greedyOutput = nextToken;
    return bytes;
  } catch (const NativeExecutionCancelled&) {
    [self finishNativeCancellation:error dispatched:dispatched]; return nil;
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return nil;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Resident forward failed" dispatched:dispatched]; return nil;
  }
}
- (nullable NSData *)prefillTokenIDs:(NSArray<NSNumber *> *)tokens expectedGeneration:(uint64_t)generation error:(NSError **)error {
  return [self prefillTokenIDs:tokens nativeConfiguration:nil expectedGeneration:generation error:error];
}
- (BOOL)configurePrefillFeatures:(NSDictionary<NSString *, id> *)configuration
                       consumer:(void (^)(NSDictionary<NSString *, id> *))consumer
             expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex);
  try {
    [self admit:generation];
    need(!_nativeExecutionPermit, "This diagnostic API is unavailable for a permission-bound owner");
    need(_position == 0 && !_features && !_cacheWindow && !_decodeOnCPU && !_requestBounded,
         "Feature observation requires a fresh full-history GPU owner");
    need([configuration isKindOfClass:NSDictionary.class] && consumer != nil,
         "Feature configuration and consumer are required");
    id rawLayers = configuration[@"layers"];
    need([rawLayers isKindOfClass:NSArray.class] && [rawLayers count] > 0,
         "Feature layer indices are required");
    backend::ResidentFeatureConfig config;
    const auto block = exactInteger(configuration[@"block_tokens"]);
    const auto pending = exactInteger(configuration[@"max_pending"]);
    need(block > 0 && block <= _prefillTokens && _prefillTokens % block == 0,
         "Feature block must divide the compiled prefill shape");
    need(pending > 0 && pending <= 8, "Feature queue limit must be1..8");
    config.block_tokens = int(block);
    int previous = -1;
    for (id item in rawLayers) {
      const auto index = exactInteger(item);
      need(index > previous && index < _layers, "Feature layers must be sorted unique valid indices");
      config.layers.push_back(int(index)); previous = int(index);
    }
    _features = std::make_shared<FeatureObserver>(config, size_t(pending), consumer, _sessionID, _weights);
    return YES;
  } catch (const std::exception& e) {
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:NO]; return NO;
  }
}
- (NSDictionary<NSString *, id> *)drainPrefillFeatures {
  std::shared_ptr<FeatureObserver> observer;
  { std::lock_guard<std::mutex> lock(_mutex); observer = _features; }
  return observer ? observer->drain() : @{@"enabled":@NO};
}
- (nullable NSData *)prefillTokenIDs:(NSArray<NSNumber *> *)tokens nativeConfiguration:(nullable NSDictionary<NSString *, id> *)configuration expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex);
  return [self prefillTokenIDsLocked:tokens nativeConfiguration:configuration policyCacheOnly:NO expectedGeneration:generation error:error];
}
// Caller holds the resident owner mutex across this complete transaction.
- (nullable NSData *)prefillTokenIDsLocked:(NSArray<NSNumber *> *)tokens nativeConfiguration:(nullable NSDictionary<NSString *, id> *)configuration policyCacheOnly:(BOOL)policyCacheOnly expectedGeneration:(uint64_t)generation error:(NSError **)error {
  BOOL dispatched = NO;
  try {
    [self admit:generation];
    [self checkNativeExecutionPermit];
    need(!_nativeExecutionPermit || configuration != nil, "Permission-bound prefill requires native target execution");
    need(!_retainedCompacted, "Retained diagnostic history permits native T1 continuation only");
    need([tokens isKindOfClass:NSArray.class], "Batch token array required");
    std::optional<backend::NativeLlamaPrefillConfig> nativeConfiguration;
    if (configuration) {
      const auto config = parseNativeConfiguration(configuration);
      need(config.architecture == backend::NativeModelArchitecture::StandardLlamaTied ||
           (!_features && !_selectionChannel && !_retainedCompacted),
           "This architecture requires full-history native prefill without advisers or sparse retention");
      need(config.layers == _layers && config.kv_heads == _heads && config.head_dim == _dim && config.vocabulary == _vocabulary && config.capacity == _capacity && !_cacheWindow,
           "Native prefill configuration differs from cache owner");
      nativeConfiguration = config;
    }
    std::unique_ptr<backend::ResidentMoEPolicyScope> policyScope;
    if (policyCacheOnly) {
      need(nativeConfiguration && backend::is_native_moe_policy_architecture(nativeConfiguration->architecture) &&
           _position == 0 && _prefillTokens == 512 && !_features && !_selectionChannel && !_core->verify,
           "Python policy prefill requires fresh Granite full-history T512");
      _core->ensurePolicyContext();
      policyScope = std::make_unique<backend::ResidentMoEPolicyScope>(_core->policyContext);
    }
    need(!_features || configuration != nil, "Feature observation requires native prefill");
    need(!_selectionChannel || configuration != nil, "Attention selection requires native prefill");
    tokens = [tokens copy];
    const size_t count = tokens.count;
    const size_t block = size_t(_prefillTokens);
    need(!_cacheWindow && count > 0 && (nativeConfiguration || count % block == 0), "Batch prefill requires full history; a partial block requires native execution");
    need(!_features || count % block == 0, "Feature observation requires complete prefill blocks");
    need(_position >= 0 && _position <= _capacity && count <= size_t(_capacity - _position), "Batch prefill exceeds capacity");
    need(!_requestBounded || (_position <= _physicalCapacity && count <= size_t(_physicalCapacity - _position)),
         "Batch prefill exceeds request allocation limit");
    need(_core->storage->length() == _position && _core->prefillBatch.empty(), "Batch prefill initial state differs");
    const auto meta = _core->prefill.method_meta("forward");
    need(meta.ok() && meta->num_instructions() == 1 && meta->num_backends() == 1 && meta->uses_backend(backend::kMLXBackendId), "Batch prefill requires one pure MLX delegate instruction");
    // Validate and own every input before the first dispatch. On any later
    // error Core (including every Invocation) is retained by quarantine.
    for (size_t offset = 0; offset < count; offset += block) {
      auto call = std::make_shared<Invocation>();
      call->tokens.reserve(block); call->positions.reserve(block);
      const size_t active = std::min(block, count - offset);
      for (size_t i = 0; i < block; ++i) {
        const auto token = i < active ? exactInteger(tokens[offset + i]) : 0;
        need(token >= 0 && token < _vocabulary, "Batch token outside vocabulary");
        call->tokens.push_back(token);
        call->positions.push_back(i < active ? int64_t(_position) + int64_t(offset + i) : 0);
      }
      call->tensors.push_back(make_tensor_ptr({1, _prefillTokens}, call->tokens.data(), ScalarType::Long));
      call->tensors.push_back(make_tensor_ptr({_prefillTokens}, call->positions.data(), ScalarType::Long));
      for (const auto& tensor : call->tensors) call->values.emplace_back(*tensor);
      _core->prefillBatch.push_back(std::move(call));
    }
    if (_selectionChannel) {
      _core->storage->set_attention_selection(nullptr);
      _activeSelection.reset();
    }
    NSData *bytes = nil;
    const bool trace_prefill = prefillTimingEnabled();
    for (size_t chunk = 0; chunk < _core->prefillBatch.size(); ++chunk) {
      auto& call = *_core->prefillBatch[chunk];
      const bool last = chunk + 1 == _core->prefillBatch.size();
      const size_t active = std::min(block, count - chunk * block);
      const int timing_position = _position + int(chunk * block);
      // ET retains its fixed ABI. Only this explicit native scope may consume
      // fewer real tokens; padding never enters the cache or output semantics.
      std::unique_ptr<backend::NativeLlamaPrefillScope> nativeScope;
      if (nativeConfiguration) {
        nativeScope = std::make_unique<backend::NativeLlamaPrefillScope>(
            _core->storage.get(), *nativeConfiguration, active < block ? int(active) : 0);
      }
      std::unique_ptr<backend::ResidentFeatureScope> featureScope;
      if (_features) {
        auto reservation = std::make_shared<FeatureReservation>(_features);
        reservation->reserved = _features->reserve();
        if (reservation->reserved) {
          auto lease = _core->prefillBatch[chunk];
          featureScope = std::make_unique<backend::ResidentFeatureScope>(
              _core->storage.get(), _features->config,
              [reservation, lease, generation](std::shared_ptr<backend::FeatureTicket> ticket) {
                need(!reservation->submitted, "Feature scope delivered multiple tickets");
                reservation->observer->submit(std::move(ticket), lease, generation);
                reservation->submitted = true;
              });
        }
      }
      [self checkNativeExecutionPermit];
      dispatched = YES;
      if (!last || policyCacheOnly) {
        backend::ResidentPrefillSubmission submission(_core->storage.get(), policyCacheOnly);
        const auto timing_start = trace_prefill
            ? PrefillTimingClock::now() : PrefillTimingClock::time_point{};
        const auto result = _core->prefill.forward(call.values);
        if (trace_prefill) prefillTiming("chunk_forward_cache_only", timing_start,
            chunk, timing_position, active, configuration != nil, last, _prefillMaxQueuedChunks);
        need(result.ok() && submission.submitted(), "Batch KV submission failed");
        // Async chunks are provisional. Cancel before another chunk or eval
        // barrier; the dedicated cancellation drain submits no lazy roots.
        [self checkNativeExecutionPermit];
        // Do not read result: cache-only forward deliberately has no logits.
      } else {
        const auto timing_start = trace_prefill
            ? PrefillTimingClock::now() : PrefillTimingClock::time_point{};
        const auto result = _core->prefill.forward(call.values);
        if (trace_prefill) prefillTiming("chunk_forward_final", timing_start,
            chunk, timing_position, active, configuration != nil, last, _prefillMaxQueuedChunks);
        need(result.ok() && result->size() == 1 && result->at(0).isTensor(), "Batch final forward failed");
        const auto& out = result->at(0).toTensor();
        need(out.scalar_type() == ScalarType::Float && out.dim() == 3 && out.size(0) == 1 && out.size(1) == 1 && out.size(2) == _vocabulary && out.strides()[2] == 1 && out.strides()[1] == _vocabulary && out.strides()[0] == _vocabulary && out.numel() == _vocabulary && out.const_data_ptr<float>(), "Batch final logits ABI differs");
        const auto drain_start = trace_prefill
            ? PrefillTimingClock::now() : PrefillTimingClock::time_point{};
        _core->storage->synchronize_storage();
        if (trace_prefill) prefillTiming("final_kv_drain", drain_start,
            chunk, timing_position, active, configuration != nil, last, _prefillMaxQueuedChunks);
        const auto* data = out.const_data_ptr<float>();
        for (int i = 0; i < _vocabulary; ++i) need(std::isfinite(data[i]), "Nonfinite batch final logits");
        bytes = [NSData dataWithBytes:data length:size_t(_vocabulary) * sizeof(float)];
        need(bytes != nil, "Could not materialize batch logits");
      }
      need(_core->storage->length() == _position + int(chunk * block + active), "Batch provisional cache length differs");
      // Bound queued work to the validated cadence (default four chunks).
      // Completion is still not published
      // until every chunk and final output has completed successfully.
      if (policyCacheOnly || (!last && (chunk + 1) % size_t(_prefillMaxQueuedChunks) == 0)) {
        const auto drain_start = trace_prefill
            ? PrefillTimingClock::now() : PrefillTimingClock::time_point{};
        _core->storage->synchronize_storage();
        if (policyCacheOnly) {
          _core->policyContext->synchronize();
          backend::clear_greedy_allocator_cache();
        }
        if (trace_prefill) prefillTiming("periodic_kv_drain", drain_start,
            chunk, timing_position, active, configuration != nil, last, _prefillMaxQueuedChunks);
      }
    }
    for (size_t chunk = 0; chunk < _core->prefillBatch.size(); ++chunk) {
      const auto& call = *_core->prefillBatch[chunk];
      const size_t active = std::min(block, count - chunk * block);
      for (size_t i = 0; i < block; ++i) {
        const int64_t token = i < active ? exactInteger(tokens[chunk * block + i]) : 0;
        const int64_t position = i < active ? int64_t(_position) + int64_t(chunk * block + i) : 0;
        need(call.tokens[i] == token && call.positions[i] == position, "Batch inputs mutated");
      }
    }
    _position += int(count); ++_generation;
    if (nativeConfiguration) {
      _core->nativePrefillConfiguration = *nativeConfiguration;
      _core->nativePrefillGeneration = _generation;
    }
    _core->prefillBatch.clear();
#ifdef TRIFUSE_ENABLE_COLD_KV
    if (_core->coldInstaller && _core->coldInstaller->encoder) {
      need(nativeConfiguration && nativeConfiguration->architecture == backend::NativeModelArchitecture::OLMoE &&
           _heads == 16 && _dim == 128 && !_core->verify && !_decodeOnCPU,
           "Cold KV experiment only supports target-only OLMoE GPU");
      _core->coldInstaller->tick(*_core->storage, [&] {
        _core->storage->synchronize_storage();
        if (_core->policyContext) _core->policyContext->synchronize();
      });
    }
#endif
    return policyCacheOnly ? [NSData data] : bytes;
  } catch (const NativeExecutionCancelled&) {
    [self finishNativeCancellation:error dispatched:dispatched]; return nil;
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    else if (_core) _core->prefillBatch.clear();
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return nil;
  } catch (...) {
    if (dispatched) [self failDispatched];
    else if (_core) _core->prefillBatch.clear();
    [self setError:error message:@"Resident batch prefill failed" dispatched:dispatched]; return nil;
  }
}
- (nullable NSDictionary<NSString *, id> *)reclaimIdleMemoryWithCacheLimitBytes:(nullable NSNumber *)limit
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  const auto started = PrefillTimingClock::now();
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation];
    [self checkNativeExecutionPermit];
    need(!_decodeOnCPU && !_features && !_core->active && _core->prefillBatch.empty() &&
         _core->storage->length() == _position, "Idle memory maintenance requires a completed GPU owner without a feature observer");
    std::optional<size_t> cacheLimit;
    if (limit) {
      const auto value = exactInteger(limit);
      need(value >= 0 && uint64_t(value) <= std::numeric_limits<size_t>::max(), "Idle cache limit must be a nonnegative byte count");
      cacheLimit = size_t(value);
    }
    [self checkNativeExecutionPermit];
    dispatched = YES;
    const auto result = backend::reclaim_idle_memory(cacheLimit);
    const double wallMS = std::chrono::duration<double, std::milli>(PrefillTimingClock::now() - started).count();
    return @{@"schema":@"idle_memory_maintenance_v1", @"sessionID":_sessionID,
      @"generationBefore":@(generation), @"generationAfter":@(_generation),
      @"absolutePositionBefore":@(_position), @"absolutePositionAfter":@(_position),
      @"synchronizedStreams":@(result.synchronized_streams),
      @"activeBytesBefore":@(result.active_before), @"activeBytesAfter":@(result.active_after),
      @"cachedBytesBefore":@(result.cached_before), @"cachedBytesAfter":@(result.cached_after),
      @"peakActiveBytesBefore":@(result.peak_before), @"peakActiveBytesAfter":@(result.peak_after),
      @"requestedCacheLimitBytes":cacheLimit ? @(*cacheLimit) : NSNull.null,
      @"previousCacheLimitBytes":result.previous_cache_limit ? @(*result.previous_cache_limit) : NSNull.null,
      @"drainMS":@(result.drain_ms), @"reclaimMS":@(result.reclaim_ms),
      @"backendWallMS":@(result.wall_ms), @"wallMS":@(wallMS),
      @"scope":@"Process-wide idle-buffer maintenance; caller quiescence required; live KV, position and generation unchanged. Cache limit is not a process-memory cap."};
  } catch (const NativeExecutionCancelled&) {
    [self finishNativeCancellation:error dispatched:dispatched]; return nil;
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return nil;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Idle memory maintenance failed" dispatched:dispatched]; return nil;
  }
}
- (nullable NSDictionary<NSString *, id> *)compactRetainingPositions:(NSArray<NSNumber *> *)positions
    nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  const auto started = PrefillTimingClock::now();
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation];
    need(!_nativeExecutionPermit, "This diagnostic API is unavailable for a permission-bound owner");
    need(!_decodeOnCPU && !_cacheWindow && !_core->verify && _verifyTokens == 0 && !_features && !_requestBounded && !_selectionChannel,
         "Diagnostic retention requires a GPU flat target with no verifier or feature observer");
    need(_position > 0 && _core->storage->length() == _position && _core->prefillBatch.empty() && !_core->active,
         "Diagnostic retention requires completed committed history");
    need(_core->nativePrefillConfiguration && _core->nativePrefillGeneration == _generation,
         "Diagnostic retention is only allowed after completed native prefill and before T1/crop/import/reset");
    const auto config = parseNativeConfiguration(configuration);
    need(config.architecture == backend::NativeModelArchitecture::StandardLlamaTied,
         "Diagnostic sparse retention is qualified only for the standard Llama adapter");
    need(config.layers == _layers && config.kv_heads == _heads && config.head_dim == _dim &&
         config.vocabulary == _vocabulary && config.capacity == _capacity &&
         sameNativeConfiguration(config, *_core->nativePrefillConfiguration),
         "Retention native configuration differs from completed prefill/cache owner");
    need([positions isKindOfClass:NSArray.class] && positions.count > 0 && positions.count <= size_t(_position),
         "Sorted original retention positions are required");
    positions = [positions copy];
    std::vector<int32_t> retained; retained.reserve(positions.count);
    int64_t previous = -1;
    for (id item in positions) {
      const auto position = exactInteger(item);
      need(position > previous && position < _position && position <= INT32_MAX,
           "Retention positions must be sorted unique within committed absolute history");
      retained.push_back(int32_t(position)); previous = position;
    }
    // All known layout/source-position admission checks happen before a GPU
    // barrier or any candidate pool construction. No default action is implied.
    _core->storage->validate_compact_retained(retained);
    const int absolutePosition = _position;
    const uint64_t generationBefore = _generation;
    dispatched = YES;
    _core->storage->synchronize_storage();
    const auto before = _core->storage->storage_stats();
    _core->storage->compact_retained(retained, mlx::core::Device(mlx::core::Device::gpu));
    const auto after = _core->storage->storage_stats();
    need(_core->storage->length() == absolutePosition && after.absolute_length == absolutePosition &&
         after.retained_tokens == int(retained.size()), "Retention changed absolute position or retained count");
    _retainedCompacted = YES; _retainedStats = after;
    ++_generation; _core->nativePrefillGeneration = _generation;
    const size_t bytesPerSlot = (_elements / size_t(_physicalCapacity)) * size_t(_kvElementBytes) * 2;
    auto stats = [&](const backend::RetainedStorageStats& value) -> NSDictionary<NSString *, id> * {
      return @{@"absoluteLength":@(value.absolute_length), @"retainedTokens":@(value.retained_tokens),
        @"allocatedSlots":@(value.payload_bytes / bytesPerSlot),
        @"allocatedPayloadBytes":@(value.payload_bytes), @"bufferBytes":@(value.buffer_bytes)};
    };
    const double elapsed = std::chrono::duration<double, std::milli>(PrefillTimingClock::now() - started).count();
    return @{@"schema":@"retained_kv_compaction_receipt_v1", @"sessionID":_sessionID,
      @"generationBefore":@(generationBefore), @"generationAfter":@(_generation),
      @"absolutePositionBefore":@(absolutePosition), @"absolutePositionAfter":@(_position),
      @"retainedPositions":positions, @"before":stats(before), @"after":stats(after), @"wallMS":@(elapsed),
      @"scope":@"Explicit diagnostic retained history; no quality, performance or memory-budget qualification.",
      @"memoryScope":@"Owned KV allocated payload includes growth slack; backing bytes are deduplicated. Neither proves allocator return to the system or process RSS reduction."};
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return nil;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Resident diagnostic retention failed" dispatched:dispatched]; return nil;
  }
}
- (BOOL)acceptPrefixLength:(NSInteger)length expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation];
    need(!_nativeExecutionPermit, "This diagnostic API is unavailable for a permission-bound owner");
    need(!_retainedCompacted, "Retained diagnostic history does not support prefix rollback");
    need(length >= 0 && length <= _position, "Accepted prefix must be within committed history");
    need(_core->storage->length() == _position, "Resident cache committed length differs");
    need(!_cacheWindow || _position - length <= 127, "Ring crop exceeds preserved rollback slack");
    // The previous forward already completed/evaluated all writes. Rewind is
    // CPU metadata publication, with no GPU barrier, K/V scan or byte copy.
    // For Ring, R=W+127 retains the pre-proposal window for <=127 rejects.
    dispatched = YES;
    need(_core->storage->rewind(int(length)), "Resident cache rejected prefix crop");
    need(_core->storage->length() == length, "Resident crop completion length differs");
    [self resetAttentionSelection];
    _position = int(length); ++_generation; return YES;
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return NO;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Resident prefix crop failed" dispatched:dispatched]; return NO;
  }
}
- (NSDictionary<NSString *, id> *)metadataUnlocked {
  const auto hot = _core ? _core->storage->attention_selection_stats() : backend::AttentionSelectionStats{};
  NSDictionary<NSString *, id> *original = @{@"sessionID":_sessionID,
    @"phase":_invalid ? @"invalid" : (_core && _core->greedyPipeline ? @"greedy_pipeline_active" : @"ready"),
    @"greedyPipeline":_greedyPipelineReceipt ? [_greedyPipelineReceipt copy] : @{@"state":@"inactive"},
    @"validLength":@(_position),
    @"nextPosition":@(_position), @"generation":@(_generation), @"capacity":@(_capacity), @"stateShape":[self shape],
    @"owner":@"MLXBackend", @"residency":@"mlx_sequence_cache", @"effectiveWeightsID":_weights,
    @"nativeExecutionPermitBound":@(_nativeExecutionPermit != nil),
    @"nativeExecutionPermitCancelled":@(_nativeExecutionPermit.isCancelled),
    @"nativeExecutionCancellationCompletion":_nativeCancellationCompletion ?: NSNull.null,
    @"vocabulary":@(_vocabulary),
    @"verifyTokens":@(_verifyTokens), @"cacheWindow":@(_cacheWindow), @"sinkTokens":@0,
    @"prefillTokens":@(_prefillTokens), @"cacheMaxWriteTokens":@(std::max(_prefillTokens, _verifyTokens)), @"prefillMaxQueuedChunks":@(_prefillMaxQueuedChunks),
    @"decodeOnCPU":@(_decodeOnCPU), @"prefillDevice":@"gpu", @"decodeDevice":_decodeOnCPU ? @"cpu" : @"gpu",
    @"deviceSelection":@"explicit_backend_stream_no_fallback",
    @"physicalCapacity":@(_physicalCapacity), @"kvStorageCapacityBytes":@(_elements * size_t(_kvElementBytes) * 2),
    @"kvAllocationPolicy":_requestBounded ? @"request_bounded_full_history_v1" : @"compiled_capacity_v1",
    @"allocationLimit":@(_physicalCapacity), @"kvPayloadBudgetBytes":_kvPayloadBudgetBytes ?: NSNull.null,
    @"kvDtype":_kvDtype, @"kvScalarType":@(_kvScalar), @"kvElementBytes":@(_kvElementBytes),
    @"cpuBLASThreadingRequested":_cpuBLASThreading,
    @"cacheApproximation":_cacheWindow ? @"recent_window_no_sinks" : (_selectionUsed ? @"indexed_attention_full_backing" : @"full_history"),
    @"attentionSelectionEnabled":@(_selectionChannel != nil),
    @"attentionSelectionEpoch":@(_selectionMailbox ? _selectionMailbox->epoch() : 0),
    @"attentionSelectionSteps":@(_selectionSteps), @"attentionFullSteps":@(_selectionFullSteps),
    @"attentionSelectionLastProposalID":_activeSelection ? @(_activeSelection->proposal_id) : NSNull.null,
    @"attentionBackingHistoryComplete":@(!_retainedCompacted && !_cacheWindow),
    @"attentionSelectionHot":@{@"reuseEnabled":@(hot.reuse_enabled),
      @"gatherLayerCalls":@(hot.gather_layer_calls), @"rebuildLayerCalls":@(hot.rebuild_layer_calls),
      @"reuseLayerCalls":@(hot.reuse_layer_calls), @"logicalCopiedElements":@(hot.copied_elements),
      @"deltaReuseEnabled":@(hot.delta_reuse_enabled), @"deltaLayerCalls":@(hot.delta_layer_calls),
      @"deltaLayout":hot.delta_reuse_enabled ? [NSString stringWithUTF8String:backend::MLXSequenceCache::attention_selection_delta_layout()] : @"disabled",
      @"canonicalReadElements":@(hot.canonical_read_elements), @"hotReadElements":@(hot.hot_read_elements),
      @"deltaReorderElements":@(hot.delta_reorder_elements),
      @"payloadBytes":@(hot.hot_payload_bytes), @"liveBytes":@(hot.hot_live_bytes), @"layers":@(hot.hot_layers),
      @"scope":@"Owner-lifetime counters; current extra hot KV payload, excluding complete canonical KV. Logical copy elements exclude zero-fill, allocator and COW traffic; not physical bandwidth or process memory."},
    @"cropSemantics":@"logical_length_only_no_gpu_dispatch_or_kv_copy"};
#ifdef TRIFUSE_ENABLE_COLD_KV
  if (_core && _core->coldInstaller && _core->coldInstaller->encoder) {
    auto *annotated = [original mutableCopy];
    const auto& c = *_core->coldInstaller;
    const auto& p = *c.policy;
    NSMutableArray *coldPerLayer=[NSMutableArray array], *hotSlots=[NSMutableArray array], *hotMax=[NSMutableArray array];
    for (int layer=0; layer<c.layers; ++layer) {
      [coldPerLayer addObject:@(_core->storage->layer_cold_prefix_length(layer))];
      [hotSlots addObject:@(_core->storage->layer_allocated_slots(layer))];
      [hotMax addObject:@(_core->storage->layer_max_slots(layer))];
    }
    annotated[@"coldKV"] = @{@"coldTokens":@(c.cold_tokens), @"blocks":@(c.blocks),
      @"coldTokensPerLayer":coldPerLayer, @"coldTokensScope":@"minimum immutable prefix across all layers",
      @"hotAllocatedSlotsPerLayer":hotSlots, @"hotMaximumSlotsPerLayer":hotMax,
      @"layeredPublicationEnabled":@(c.layered), @"directOutputEnabled":@(c.layered),
      @"layerPublications":@(c.layer_publications),
      @"unpublishedCandidateBytes":@(c.candidate_bytes()), @"maxUnpublishedCandidateBytes":@(c.max_candidate_bytes),
      @"oldColdPrefixCopyBytes":@(c.old_prefix_copy_bytes),
      @"publications":@(c.publications), @"pending":@(bool(c.job)),
      @"requestPublications":@(c.request_publications),
      @"lastTriggerPosition":@(c.last_trigger_position),
      @"lastCompletedPosition":@(c.last_completed_position),
      @"postMaintenanceScope":@"Current fresh request positions; whole-job completion after every layer published. No added GPU work or tensor references.",
      @"pendingCompletedBlocks":@(c.job ? c.job->completed : 0),
      @"pendingTotalBlocks":@(c.job ? c.job->total() : 0),
      @"thresholdBytes":@(p.effectiveHighBytes()), @"initialThresholdBytes":@(p.config().high_bytes),
      @"adaptive":@(p.config().adaptive), @"thresholdEvaluations":@(p.stats().evaluations),
      @"thresholdTriggers":@(p.stats().triggers), @"thresholdIncreases":@(p.stats().threshold_increases),
      @"transactionWallMS":@(c.transaction_ms), @"snapshotMS":@(c.snapshot_ms),
      @"snapshotWaitMS":@(c.snapshot_wait_ms), @"publishMS":@(c.publish_ms), @"collectMS":@(c.collect_ms),
      @"snapshotDeferrals":@(c.snapshot_deferrals), @"snapshotDrains":@(c.snapshot_drains),
      @"hostDeltaStagingBytes":@(c.delta_bytes()), @"maxHostDeltaStagingBytes":@(c.max_delta_bytes),
      @"encodeTimingScope":@"collected blocks only; pending/uncollected work excluded",
      @"transactionWallScope":@"overlapped job span, not additive request cost",
      @"maxSnapshotBytes":@(c.max_snapshot_bytes), @"snapshotBatchBlocks":@(c.snapshot_batch),
      @"ticksWhileEncoding":@(c.ticks_while_encoding), @"discardedJobs":@(c.discarded_jobs),
      @"resetWaitMS":@(c.reset_wait_ms), @"encodeMS":@(c.encode_ms),
      @"predictMS":@(c.predict_ms), @"packMS":@(c.pack_ms), @"originalPayloadBytes":@(c.before_bytes),
      @"publishedPayloadBytes":@(c.after_bytes), @"mlxActiveBefore":@(c.active_before), @"mlxActiveAfter":@(c.active_after),
      @"encoderPreparationMS":@(c.encoder->preparation().total_ms),
      @"actualANEVerified":@NO, @"cpuFallbackAllowed":@(!c.encoder->preparation().cpu_only),
      @"encoderComputeUnits":c.encoder->preparation().cpu_only ? @"cpuOnly" : @"cpuAndNeuralEngine",
      @"backgroundEncodingEnabled":@YES,
      @"workerQoSRequested":@"utility", @"codecFormat":@"mlx_affine_uint32_g64_b8",
      @"coldAttention":@"MLX quantized_matmul + global precise softmax + quantized_matmul; hot FP16 matmul",
      @"hadamardTransforms":@NO, @"snapshotWaitIsSubsetOfSnapshotMS":@YES,
      @"scope":@"KV allocation threshold; bounded owner snapshots; one asynchronous codec; selected whole-job or per-layer atomic publication; pending work stays with cache; host overlap counters do not prove physical ANE execution"};
    original = annotated;
  }
#endif
  if (_core && _core->storage->has_cold_history()) {
    auto *mixed = [original mutableCopy];
    mixed[@"cacheApproximation"] = @"full_positions_affine8_cold_fp16_hot";
    mixed[@"kvDtypeSemantics"] = @"compiled_prefill_dtype_only_actual_storage_is_mixed";
    mixed[@"actualKVStorage"] = @"cold_mlx_affine_uint32_g64_b8_fp16_scales_biases_hot_float16";
    mixed[@"cropSemantics"] = @"unsupported_for_cold_history";
    mixed[@"physicalCapacitySemantics"] = @"original_logical_request_limit_not_hot_pool_allocation";
    mixed[@"stateShapeSemantics"] = @"logical_geometry_not_physical_mixed_representation";
    mixed[@"kvStorageCapacityBytesSemantics"] = @"original_FP16_capacity_not_current_allocation";

    return mixed;
  }
  if (!_retainedCompacted) return original;
  NSMutableDictionary<NSString *, id> *state = [original mutableCopy];
  if (_retainedCompacted) {
    const size_t bytesPerSlot = (_elements / size_t(_physicalCapacity)) * size_t(_kvElementBytes) * 2;
    state[@"cacheApproximation"] = @"explicit_retained_history";
    state[@"cacheLayout"] = @"absolute_positions_dense_retained_slots_v1";
    state[@"cropSemantics"] = @"unsupported_for_retained_diagnostic_history";
    state[@"physicalCapacitySemantics"] = @"original_configured_cap_not_current_allocation";
    state[@"kvStorageCapacityBytesSemantics"] = @"original_configured_capacity_not_current_allocation";
    state[@"stateShapeSemantics"] = @"original_configured_geometry_not_retained_storage_shape";
    state[@"originalPhysicalCapacity"] = @(_physicalCapacity);
    state[@"originalKVStorageCapacityBytes"] = @(_elements * size_t(_kvElementBytes) * 2);
    state[@"currentRetainedTokens"] = @(_retainedStats.retained_tokens);
    state[@"currentAllocatedSlots"] = @(_retainedStats.payload_bytes / bytesPerSlot);
    state[@"currentKVAllocatedPayloadBytes"] = @(_retainedStats.payload_bytes);
    state[@"currentKVBufferBytes"] = @(_retainedStats.buffer_bytes);
    state[@"retainedStorageScope"] = _invalid ? @"last_completed_statistics_before_invalid_completion" : @"completed_owned_KV_storage_not_process_RSS_or_allocator_system_return";
  }
  return state;
}
- (nullable NSDictionary<NSString *, id> *)inspectMetadata:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex);
  return [self metadataUnlocked];
}
- (nullable NSDictionary<NSString *, id> *)inspect:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex);
  if (_core && _core->greedyPipeline) {
    [self setError:error message:@"Owned-storage inspection requires a drained greedy pipeline" dispatched:NO];
    return nil;
  }
  // Global counters must only be sampled by a caller that has drained model
  // execution. This owner lock cannot protect another model's allocator writes.
  NSMutableDictionary<NSString *, id> *state = [[self metadataUnlocked] mutableCopy];
  state[@"mlxActiveBytes"] = @(mlx::core::get_active_memory());
  state[@"mlxCachedBytes"] = @(mlx::core::get_cache_memory());
  state[@"mlxPeakActiveBytes"] = @(mlx::core::get_peak_memory());
  if (!_invalid) {
    try {
      const auto owned = _core->storage->storage_stats();
      const size_t bytesPerSlot = (_elements / size_t(_physicalCapacity)) * size_t(_kvElementBytes) * 2;
      state[@"currentAllocatedSlots"] = _core->storage->has_cold_history() ? (id)NSNull.null : @(owned.payload_bytes / bytesPerSlot);
      state[@"coldKVPayloadBytes"] = @(owned.cold_payload_bytes);
      state[@"hotKVPayloadBytes"] = @(owned.hot_payload_bytes);
      state[@"originalFP16OwnerBytes"] = @(owned.original_fp16_owner_bytes);
      state[@"currentKVAllocatedPayloadBytes"] = @(owned.payload_bytes);
      state[@"currentKVBufferBytes"] = @(owned.buffer_bytes);
      state[@"currentKVStorageScope"] = @"completed_owned_storage_only_excludes_growth_temporaries_weights_workspace_and_process_peak";
    } catch (const std::exception& e) {
      [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:NO]; return nil;
    }
  }
  return state;
}
- (nullable NSDictionary<NSString *, id> *)exportStateExpectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation];
    need(!_nativeExecutionPermit, "This diagnostic API is unavailable for a permission-bound owner");
    need(!_selectionChannel && !_requestBounded && !_retainedCompacted && !_cacheWindow && _kvScalar == int(ScalarType::Float), "Host FP32 snapshot requires unbounded uncompacted flat FP32 storage without attention selection"); dispatched = YES;
    diagnosticMemory("full_snapshot_begin");
    const auto state = _core->storage->export_flat_fp32(_position);
    diagnosticMemory("full_snapshot_vectors_ready");
    return @{@"keys":[NSData dataWithBytes:state.keys.data() length:_elements * sizeof(float)],
      @"values":[NSData dataWithBytes:state.values.data() length:_elements * sizeof(float)],
      @"shape":[self shape], @"validLength":@(_position), @"generation":@(_generation),
      @"sessionID":_sessionID, @"effectiveWeightsID":_weights, @"layout":@"flat_bhcd_fp32"};
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return nil;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Resident export failed" dispatched:dispatched]; return nil;
  }
}
- (BOOL)importState:(NSDictionary<NSString *, id> *)state expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation];
    need(!_nativeExecutionPermit, "This diagnostic API is unavailable for a permission-bound owner");
    need(!_selectionChannel && !_requestBounded && !_retainedCompacted && !_cacheWindow && _kvScalar == int(ScalarType::Float), "Host FP32 snapshot requires unbounded uncompacted flat FP32 storage without attention selection");
    need([state isKindOfClass:NSDictionary.class], "Snapshot dictionary required");
    need([state[@"effectiveWeightsID"] isKindOfClass:NSString.class] && [state[@"effectiveWeightsID"] isEqualToString:_weights] && [state[@"layout"] isEqual:@"flat_bhcd_fp32"], "Snapshot weights/layout differ");
    NSArray *shape = state[@"shape"], *expected = [self shape];
    need([shape isKindOfClass:NSArray.class] && shape.count == expected.count, "Snapshot shape differs");
    for (NSUInteger i = 0; i < shape.count; ++i) need(exactInteger(shape[i]) == exactInteger(expected[i]), "Snapshot shape differs");
    const int64_t length = exactInteger(state[@"validLength"]); need(length >= 0 && length <= _capacity, "Snapshot length outside capacity");
    NSData *keys = state[@"keys"], *values = state[@"values"];
    need([keys isKindOfClass:NSData.class] && [values isKindOfClass:NSData.class] && keys.length == _elements * 4 && values.length == _elements * 4, "Snapshot storage size differs");
    auto input = std::make_unique<backend::FlatFP32State>();
    input->layers = _layers; input->heads = _heads; input->head_dim = _dim; input->capacity = _capacity; input->valid_length = int(length);
    input->keys.resize(_elements); input->values.resize(_elements);
    std::memcpy(input->keys.data(), keys.bytes, keys.length); std::memcpy(input->values.data(), values.bytes, values.length);
    for (const auto *array : {&input->keys, &input->values}) for (float x : *array) need(std::isfinite(x), "Snapshot contains nonfinite values");
    _core->importing = std::move(input); dispatched = YES;
    _core->storage->import_flat_fp32(*_core->importing); _core->storage->synchronize_flat();
    need(_core->storage->length() == length, "Imported cache length differs");
    [self resetAttentionSelection];
    _position = int(length); ++_generation; _core->importing.reset(); return YES;
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return NO;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Resident import failed" dispatched:dispatched]; return NO;
  }
}
- (BOOL)drainAcceptedMaintenanceExpectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation];
    [self checkNativeExecutionPermit];
    need(!_core->greedyPipeline && !_core->active && _core->prefillBatch.empty(),
         "Maintenance drain requires completed foreground inference");
#ifdef TRIFUSE_ENABLE_COLD_KV
    if (_core->coldInstaller) {
      dispatched = YES;
      _core->coldInstaller->drain_accepted(*_core->storage, [&] {
        // A completed codec block may publish a layer here. Cancellation can
        // happen while polling, so admission must be checked at publication,
        // not only at entry to this waiting operation.
        [self checkNativeExecutionPermit];
        _core->storage->synchronize_storage();
      });
    }
#endif
    return YES;
  } catch (const NativeExecutionCancelled&) {
    [self finishNativeCancellation:error dispatched:dispatched]; return NO;
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return NO;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Maintenance drain failed" dispatched:dispatched]; return NO;
  }
}

- (BOOL)beginRequestWithAllocationLimit:(NSInteger)allocationLimit
    kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation];
    [self checkNativeExecutionPermit];
    need(!_decodeOnCPU && !_cacheWindow && !_core->verify && _verifyTokens == 0 &&
         !_features && !_retainedCompacted && !_core->active &&
         _core->prefillBatch.empty() && !_core->importing,
         "New request requires idle full-history GPU target without verifier or observer");
    need(_core->storage->length() == _position, "Resident cache position differs");
    need(allocationLimit >= 0 && allocationLimit <= _capacity, "Request allocation limit outside compiled capacity");
    need(!kvPayloadBudgetBytes || allocationLimit > 0, "KV payload budget requires request-bound allocation");
    const std::optional<int> ceiling = allocationLimit > 0
        ? std::optional<int>(int(allocationLimit)) : std::nullopt;
    _core->storage->validate_request_allocation(ceiling);
    const int physicalCapacity = ceiling.value_or(_capacity);
    size_t elements = 1;
    for (int x : {_layers, _heads, physicalCapacity, _dim}) {
      need(elements <= SIZE_MAX / size_t(x), "Resident state size overflow"); elements *= size_t(x);
    }
    need(elements <= SIZE_MAX / (2 * size_t(_kvElementBytes)), "Resident paired-state byte size overflow");
    if (kvPayloadBudgetBytes) {
      const auto budget = exactInteger(kvPayloadBudgetBytes);
      need(budget > 0 && uint64_t(budget) >= elements * size_t(_kvElementBytes) * 2,
           "KV payload budget cannot hold complete requested history");
    }
    NSNumber *newBudget = [kvPayloadBudgetBytes copy];
    [self checkNativeExecutionPermit];
    dispatched = YES;
    _core->storage->synchronize_storage();
    [self checkNativeExecutionPermit];
#ifdef TRIFUSE_ENABLE_COLD_KV
    if (_core->coldInstaller) _core->coldInstaller->reset();
#endif
    _core->storage->reset_request_allocation(ceiling);
    need(_core->storage->length() == 0 && !_core->storage->has_compacted_history(),
         "New request did not clear old cache history");
    _core->nativePrefillConfiguration.reset(); _core->nativePrefillGeneration = 0;
    _physicalCapacity = physicalCapacity; _elements = elements;
    _requestBounded = ceiling.has_value(); _kvPayloadBudgetBytes = newBudget;
    _retainedStats = {}; _position = 0; ++_generation;
    [self resetAttentionSelection];
    _selectionSteps = 0; _selectionFullSteps = 0; _selectionUsed = NO;
    return YES;
  } catch (const NativeExecutionCancelled&) {
    [self finishNativeCancellation:error dispatched:dispatched]; return NO;
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return NO;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Resident begin request failed" dispatched:dispatched]; return NO;
  }
}
- (BOOL)retainPrefixLength:(NSInteger)length allocationLimit:(NSInteger)allocationLimit
    kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes
    expectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation];
    [self checkNativeExecutionPermit];
    need(!_decodeOnCPU && !_cacheWindow && !_core->verify && _verifyTokens == 0 &&
         !_features && !_retainedCompacted && !_selectionChannel && !_selectionUsed && !_core->active &&
         _core->prefillBatch.empty() && !_core->importing,
         "Prefix reuse requires idle full-history GPU target without verifier or observer");
    need(_core->storage->length() == _position, "Resident cache position differs");
    need(allocationLimit >= 0 && allocationLimit <= _capacity, "Request allocation limit outside compiled capacity");
    need(!kvPayloadBudgetBytes || allocationLimit > 0, "KV payload budget requires request-bound allocation");
    const std::optional<int> ceiling = allocationLimit > 0
        ? std::optional<int>(int(allocationLimit)) : std::nullopt;
    need(length >= 0 && length <= _position, "Retained prefix outside completed history");
    _core->storage->validate_prefix_allocation(int(length), ceiling);
    const int physicalCapacity = ceiling.value_or(_capacity);
    size_t elements = 1;
    for (int x : {_layers, _heads, physicalCapacity, _dim}) {
      need(elements <= SIZE_MAX / size_t(x), "Resident state size overflow"); elements *= size_t(x);
    }
    need(elements <= SIZE_MAX / (2 * size_t(_kvElementBytes)), "Resident paired-state byte size overflow");
    if (kvPayloadBudgetBytes) {
      const auto budget = exactInteger(kvPayloadBudgetBytes);
      need(budget > 0 && uint64_t(budget) >= elements * size_t(_kvElementBytes) * 2,
           "KV payload budget cannot hold complete requested history");
    }
    NSNumber *newBudget = [kvPayloadBudgetBytes copy];
    [self checkNativeExecutionPermit];
    dispatched = YES;
    _core->storage->synchronize_storage();
    [self checkNativeExecutionPermit];
#ifdef TRIFUSE_ENABLE_COLD_KV
    if (_core->coldInstaller) _core->coldInstaller->reset();
#endif
    _core->storage->retain_prefix_allocation(int(length), ceiling);
    need(_core->storage->length() == length && !_core->storage->has_compacted_history(),
         "Prefix allocation did not preserve the requested dense history");
    _core->nativePrefillConfiguration.reset(); _core->nativePrefillGeneration = 0;
    _physicalCapacity = physicalCapacity; _elements = elements;
    _requestBounded = ceiling.has_value(); _kvPayloadBudgetBytes = newBudget;
    _retainedStats = {}; _position = int(length); ++_generation;
    [self resetAttentionSelection];
    _selectionSteps = 0; _selectionFullSteps = 0; _selectionUsed = NO;
    return YES;
  } catch (const NativeExecutionCancelled&) {
    [self finishNativeCancellation:error dispatched:dispatched]; return NO;
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return NO;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Resident prefix allocation failed" dispatched:dispatched]; return NO;
  }
}
- (BOOL)resetExpectedGeneration:(uint64_t)generation error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_mutex); BOOL dispatched = NO;
  try {
    [self admit:generation];
    [self checkNativeExecutionPermit]; dispatched = YES;
    _core->storage->synchronize_storage();
    [self checkNativeExecutionPermit];
#ifdef TRIFUSE_ENABLE_COLD_KV
    if (_core->coldInstaller) _core->coldInstaller->reset();
#endif
    _core->storage->reset_storage();
    need(_core->storage->length() == 0, "Resident reset did not clear state");
    need(!_core->storage->has_compacted_history(), "Resident reset did not clear retained layout");
    _retainedCompacted = NO; _retainedStats = {};
    [self resetAttentionSelection];
    _selectionSteps = 0; _selectionFullSteps = 0; _selectionUsed = NO;
    _position = 0; ++_generation; return YES;
  } catch (const NativeExecutionCancelled&) {
    [self finishNativeCancellation:error dispatched:dispatched]; return NO;
  } catch (const std::exception& e) {
    if (dispatched) [self failDispatched];
    [self setError:error message:[NSString stringWithUTF8String:e.what()] dispatched:dispatched]; return NO;
  } catch (...) {
    if (dispatched) [self failDispatched];
    [self setError:error message:@"Resident reset failed" dispatched:dispatched]; return NO;
  }
}
- (void)invalidateWithReason:(NSString *)reason {
  std::lock_guard<std::mutex> lock(_mutex);
  if (_invalid) return;
  // Only an explicitly observed pre-dispatch cancellation establishes this
  // path. A cancelled permit alone says nothing about execution completion.
  if (_nativeCancellationKnownIdle &&
      [_nativeCancellationCompletion isEqualToString:@"not_dispatched"] &&
      _core && !_core->active && _core->prefillBatch.empty() && !_core->importing &&
      _core->storage->length() == _position) {
    _invalid = YES;
    try {
      backend::drain_submitted_work();
      _core.reset();
      return;
    } catch (...) {
      // Preserve all buffers if the completion barrier itself is uncertain.
      _nativeCancellationCompletion = @"unknown";
      _nativeCancellationKnownIdle = NO;
    }
  }
  [self failDispatched];
}
@end
