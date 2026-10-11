#include <pthread.h>
#include "ColdKVEncoder.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <os/signpost.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {
using Clock = std::chrono::steady_clock;
uint64_t nowNS() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now().time_since_epoch()).count();
}
double milliseconds(uint64_t start, uint64_t end) { return double(end - start) / 1e6; }
std::runtime_error failure(const char* stage, NSError* error) {
  const char* description = error.localizedDescription.UTF8String;
  return std::runtime_error(std::string(stage) + ": " + (description ? description : "unknown CoreML error"));
}
std::runtime_error failure(NSException* error) {
  const char* description = error.reason.UTF8String;
  return std::runtime_error(std::string("CoreML exception: ") + (description ? description : "unknown"));
}
float fromHalf(uint16_t bits) {
  static_assert(sizeof(_Float16) == sizeof(uint16_t));
  _Float16 value;
  std::memcpy(&value, &bits, sizeof(value));
  return static_cast<float>(value);
}
struct Plane {
  size_t row_stride;
  size_t column_stride;
};
Plane checkedPlane(const void* bytes, NSInteger count, NSArray<NSNumber*>* strides, size_t rows) {
  constexpr size_t columns = ColdKVEncoder::kVectors;
  if (!bytes || count < 0 || count % 2 || strides.count != 4 ||
      strides[1].longLongValue <= 0 || strides[3].longLongValue <= 0) {
    throw std::runtime_error("Invalid FP16 cold-KV buffer/strides");
  }
  const size_t row = strides[1].unsignedLongLongValue;
  const size_t column = strides[3].unsignedLongLongValue;
  const size_t elements = static_cast<size_t>(count) / 2;
  if (!elements || rows - 1 > (elements - 1) / row ||
      columns - 1 > (elements - 1 - (rows - 1) * row) / column) {
    throw std::runtime_error("Cold-KV strides exceed buffer bounds");
  }
  return {row, column};
}
os_log_t codecLog() {
  static os_log_t log = os_log_create("com.zhuanz.trifuse", "ColdKVEncoder");
  return log;
}
void validateSnapshot(const std::shared_ptr<const ColdKVEncoder::Snapshot>& snapshot) {
  if (!snapshot || snapshot->keys.size() != ColdKVEncoder::kElements ||
      snapshot->values.size() != ColdKVEncoder::kElements ||
      snapshot->identity.layer < 0 || snapshot->identity.start_position < 0 ||
      snapshot->identity.start_position > std::numeric_limits<int>::max() - int(ColdKVEncoder::kTokens)) {
    throw std::invalid_argument("ColdKVEncoder expects owning finite-block geometry [16,256,128]");
  }
  ColdKVEncoder::validateOutputTarget(*snapshot);
}
} // namespace

struct ColdKVEncoder::Impl {
  MLModel* model = nil;
  MLMultiArray* input = nil;
  MLDictionaryFeatureProvider* provider = nil;
  MLPredictionOptions* options = nil;
  NSURL* ownedCompiledURL = nil;
  Preparation prepared;
  std::mutex mutex;
  std::condition_variable condition;
  std::thread worker;
  bool stopping = false;
  bool outstanding = false;
  bool ready = false;
  std::shared_ptr<const Snapshot> pending;
  uint64_t pendingSubmitted = 0;
  std::shared_ptr<const Result> result;
  std::exception_ptr error;

  ~Impl() {
    stop();
    @autoreleasepool {
      options = nil; provider = nil; input = nil; model = nil;
      if (ownedCompiledURL) {
        [[NSFileManager defaultManager] removeItemAtURL:ownedCompiledURL error:nil];
      }
    }
  }

  void stop() {
    if (worker.joinable() && std::this_thread::get_id() == worker.get_id()) {
      throw std::logic_error("ColdKVEncoder finish must run on owner, not worker");
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    condition.notify_all();
    if (worker.joinable()) worker.join();
  }

  Poll takeLocked() {
    if (ready) {
      Poll value{error ? State::Failed : State::Ready, std::move(result), std::move(error)};
      ready = false;
      outstanding = false;
      return value;
    }
    return {outstanding ? State::Busy : stopping ? State::Closed : State::Idle, nullptr, nullptr};
  }

  void encodeTensor(const std::vector<uint16_t>& source, Quantized& destination,
                    const QuantizedTarget* direct, TensorTiming& timing) {
    const uint64_t begin = nowNS();
    QuantizedTarget target;
    if (direct) target = *direct;
    else {
      destination.data.resize(kVectors * kWordsPerVector);
      destination.scales.resize(kVectors * kGroupsPerVector);
      destination.biases.resize(kVectors * kGroupsPerVector);
      target.data = destination.data.data();
      target.scales = destination.scales.data();
      target.biases = destination.biases.data();
      target.data_capacity_words = destination.data.size();
      target.scales_capacity_groups = destination.scales.size();
      target.biases_capacity_groups = destination.biases.size();
    }
    const uint64_t copyBegin = nowNS();
    std::exception_ptr callbackError;
    auto* slot = &callbackError;
    const auto* sourcePointer = &source;
    [input getMutableBytesWithHandler:^(void* bytes, NSInteger count, NSArray<NSNumber*>* strides) {
      try {
        const auto layout = checkedPlane(bytes, count, strides, kDimensions);
        auto* output = static_cast<uint8_t*>(bytes);
        for (size_t channel = 0; channel < kDimensions; ++channel) {
          for (size_t vector = 0; vector < kVectors; ++vector) {
            const uint16_t bits = (*sourcePointer)[vector * kDimensions + channel];
            if (!std::isfinite(fromHalf(bits)))
              throw std::runtime_error("Nonfinite cold-KV affine input");
            const size_t index = channel * layout.row_stride + vector * layout.column_stride;
            std::memcpy(output + index * sizeof(bits), &bits, sizeof(bits));
          }
        }
      } catch (...) { *slot = std::current_exception(); }
    }];
    timing.input_copy_ms = milliseconds(copyBegin, nowNS());
    if (callbackError) std::rethrow_exception(callbackError);
    timing.logical_input_bytes = kElements * sizeof(uint16_t);

    NSError* predictionError = nil;
    id<MLFeatureProvider> prediction = nil;
    const auto log = codecLog();
    const auto signpost = os_signpost_id_generate(log);
    const uint64_t predictBegin = nowNS();
    os_signpost_interval_begin(log, signpost, "ColdKVAffineBounds");
    @try {
      prediction = [model predictionFromFeatures:provider options:options error:&predictionError];
    } @finally {
      timing.predict_ms = milliseconds(predictBegin, nowNS());
      os_signpost_interval_end(log, signpost, "ColdKVAffineBounds");
    }
    if (!prediction) throw failure("Predict cold-KV affine bounds", predictionError);
    const uint64_t outputBegin = nowNS();
    MLMultiArray* values = [prediction featureValueForName:@"bounds"].multiArrayValue;
    if (!values || values.dataType != MLMultiArrayDataTypeFloat16 ||
        ![values.shape isEqualToArray:@[@1,@4,@1,@4096]]) {
      throw std::runtime_error("Cold-KV prediction output shape or dtype differs");
    }
    callbackError = nullptr;
    [values getBytesWithHandler:^(const void* bytes, NSInteger count) {
      try {
        const auto layout = checkedPlane(bytes, count, values.strides, 4);
        const auto* base = static_cast<const uint8_t*>(bytes);
        for (size_t vector = 0; vector < kVectors; ++vector) {
          for (size_t group = 0; group < kGroupsPerVector; ++group) {
            uint16_t minimum, maximum;
            const size_t minimum_index = group * layout.row_stride + vector * layout.column_stride;
            const size_t maximum_index = (group + 2) * layout.row_stride + vector * layout.column_stride;
            std::memcpy(&minimum, base + minimum_index * sizeof(minimum), sizeof(minimum));
            std::memcpy(&maximum, base + maximum_index * sizeof(maximum), sizeof(maximum));
            ColdKVEncoder::packAffineGroup64ToTarget(
                sourcePointer->data() + vector * kDimensions + group * kGroupSize,
                minimum, maximum, target, vector, group);
          }
        }
      } catch (...) { *slot = std::current_exception(); }
    }];
    timing.output_read_pack_ms = milliseconds(outputBegin, nowNS());
    if (callbackError) std::rethrow_exception(callbackError);
    timing.logical_output_read_bytes = 4 * kVectors * sizeof(uint16_t);
    timing.stored_bytes = kVectors * (kWordsPerVector * sizeof(uint32_t) +
        2 * kGroupsPerVector * sizeof(uint16_t));
    timing.total_ms = milliseconds(begin, nowNS());
  }

  std::shared_ptr<const Result> execute(const Snapshot& snapshot, uint64_t submitted) {
    @autoreleasepool {
      @try {
        auto value = std::make_shared<Result>();
        value->identity = snapshot.identity;
        value->direct_output = bool(snapshot.output);
        value->submitted_monotonic_ns = submitted;
        value->started_monotonic_ns = nowNS();
        value->queue_wait_ms = milliseconds(submitted, value->started_monotonic_ns);
        encodeTensor(snapshot.keys, value->keys,
            snapshot.output ? &snapshot.output->keys : nullptr, value->key_timing);
        encodeTensor(snapshot.values, value->values,
            snapshot.output ? &snapshot.output->values : nullptr, value->value_timing);
        value->ended_monotonic_ns = nowNS();
        value->encode_ms = milliseconds(value->started_monotonic_ns, value->ended_monotonic_ns);
        return value;
      } @catch (NSException* exception) { throw failure(exception); }
    }
  }

  void loop() {
    // Auxiliary work yields CPU priority to token generation; CoreML still
    // chooses CPU/ANE placement. This is not a hardware isolation guarantee.
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    for (;;) {
      std::shared_ptr<const Snapshot> snapshot;
      uint64_t submitted;
      {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return stopping || bool(pending); });
        if (!pending) return;
        snapshot = std::move(pending);
        submitted = pendingSubmitted;
      }
      std::shared_ptr<const Result> completed;
      std::exception_ptr failureValue;
      try { completed = execute(*snapshot, submitted); }
      catch (...) { failureValue = std::current_exception(); }
      snapshot.reset(); // finish all scoped CoreML accesses before releasing input
      {
        std::lock_guard<std::mutex> lock(mutex);
        result = std::move(completed);
        error = std::move(failureValue);
        ready = true;
        if (error) stopping = true; // visibly close after error; no implicit retry
      }
      condition.notify_all();
    }
  }
};

ColdKVEncoder::ColdKVEncoder(const std::string& path) : impl_(std::make_unique<Impl>()) {
  const uint64_t began = nowNS();
  @autoreleasepool {
    @try {
      if (path.empty()) throw std::invalid_argument("Cold-KV model path is empty");
      NSString* string = [[NSString alloc] initWithBytes:path.data() length:path.size() encoding:NSUTF8StringEncoding];
      if (!string) throw std::invalid_argument("Cold-KV path must be UTF8");
      NSURL* url = [NSURL fileURLWithPath:string];
      if (![[NSFileManager defaultManager] fileExistsAtPath:url.path]) throw std::runtime_error("Cold-KV package missing");
      NSError* error = nil;
      if (![url.pathExtension.lowercaseString isEqualToString:@"mlmodelc"]) {
        const uint64_t start = nowNS();
        url = [MLModel compileModelAtURL:url error:&error];
        if (!url) throw failure("Compile cold-KV model", error);
        impl_->ownedCompiledURL = url;
        impl_->prepared.compiled_package = true;
        impl_->prepared.compile_ms = milliseconds(start, nowNS());
      }
      const uint64_t loadStart = nowNS();
      MLModelConfiguration* configuration = [[MLModelConfiguration alloc] init];
      const char* cpuOnly = std::getenv("TRIFUSE_COLD_KV_CPU_ONLY");
      if (cpuOnly && std::strcmp(cpuOnly, "0") && std::strcmp(cpuOnly, "1"))
        throw std::invalid_argument("TRIFUSE_COLD_KV_CPU_ONLY must be 0 or 1");
      impl_->prepared.cpu_only = cpuOnly && !std::strcmp(cpuOnly, "1");
      configuration.computeUnits = impl_->prepared.cpu_only ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndNeuralEngine;
      impl_->model = [MLModel modelWithContentsOfURL:url configuration:configuration error:&error];
      if (!impl_->model) throw failure("Load cold-KV model", error);
      MLModelDescription* description = impl_->model.modelDescription;
      NSDictionary* metadata = description.metadata[MLModelCreatorDefinedKey];
      if (![metadata[@"trifuse.component"] isEqual:@"affine_bounds"] ||
          ![metadata[@"trifuse.format"] isEqual:@"mlx_affine_uint32_g64_b8"] ||
          ![metadata[@"trifuse.group_size"] isEqual:@"64"] ||
          ![metadata[@"trifuse.bits"] isEqual:@"8"] ||
          ![metadata[@"trifuse.trained"] isEqual:@"false"]) {
        throw std::runtime_error("Expected untrained MLX g64/b8 affine_bounds package metadata");
      }
      const auto shape = @[@1, @128, @1, @4096];
      const auto boundsShape = @[@1, @4, @1, @4096];
      MLFeatureDescription* x = description.inputDescriptionsByName[@"frozen_state"];
      MLFeatureDescription* y = description.outputDescriptionsByName[@"bounds"];
      if (description.inputDescriptionsByName.count != 1 || description.outputDescriptionsByName.count != 1 ||
          !x || !y || x.type != MLFeatureTypeMultiArray || y.type != MLFeatureTypeMultiArray ||
          x.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16 ||
          y.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16 ||
          ![x.multiArrayConstraint.shape isEqualToArray:shape] || ![y.multiArrayConstraint.shape isEqualToArray:boundsShape]) {
        throw std::runtime_error("Expected FP16 frozen_state [1,128,1,4096] / bounds [1,4,1,4096]");
      }
      impl_->input = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:&error];
      if (!impl_->input) throw failure("Allocate cold-KV input", error);
      impl_->provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:
          @{@"frozen_state": [MLFeatureValue featureValueWithMultiArray:impl_->input]} error:&error];
      if (!impl_->provider) throw failure("Create cold-KV provider", error);
      impl_->options = [[MLPredictionOptions alloc] init];
      impl_->prepared.load_and_input_allocation_ms = milliseconds(loadStart, nowNS());
    } @catch (NSException* exception) { throw failure(exception); }
  }
  auto* instance = impl_.get();
  impl_->worker = std::thread([instance] { instance->loop(); });
  impl_->prepared.total_ms = milliseconds(began, nowNS());
}

ColdKVEncoder::~ColdKVEncoder() = default;
const ColdKVEncoder::Preparation& ColdKVEncoder::preparation() const { return impl_->prepared; }

bool ColdKVEncoder::trySubmit(std::shared_ptr<const Snapshot> snapshot) {
  validateSnapshot(snapshot);
  std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock() || impl_->stopping || impl_->outstanding) return false;
  impl_->pending = std::move(snapshot);
  impl_->pendingSubmitted = nowNS();
  impl_->outstanding = true;
  lock.unlock();
  impl_->condition.notify_one();
  return true;
}

ColdKVEncoder::Poll ColdKVEncoder::poll() {
  std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) return {State::Busy, nullptr, nullptr};
  return impl_->takeLocked();
}

std::shared_ptr<const ColdKVEncoder::Result> ColdKVEncoder::encode(std::shared_ptr<const Snapshot> snapshot) {
  validateSnapshot(snapshot);
  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (impl_->stopping || impl_->outstanding) throw std::logic_error("ColdKVEncoder encode requires an idle open worker");
  impl_->pending = std::move(snapshot);
  impl_->pendingSubmitted = nowNS();
  impl_->outstanding = true;
  impl_->condition.notify_one();
  impl_->condition.wait(lock, [&] { return impl_->ready; });
  auto value = impl_->takeLocked();
  lock.unlock();
  if (value.error) std::rethrow_exception(value.error);
  if (!value.result) throw std::runtime_error("ColdKVEncoder completed without a result");
  return std::move(value.result);
}

void ColdKVEncoder::finish() { impl_->stop(); }
