#pragma once

#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/runtime/core/data_loader.h>

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <utility>

// One immutable external file per resident owner. Module wrappers share exact
// segment mappings; outstanding FreeableBuffers own their mapping independently.
class TFETSharedSegmentDataLoader final : public executorch::runtime::DataLoader {
  using Error = executorch::runtime::Error;
  using Buffer = executorch::runtime::FreeableBuffer;
  template <class T> using Result = executorch::runtime::Result<T>;
  using Mmap = executorch::extension::MmapDataLoader;
  struct Mapping {
    explicit Mapping(Buffer&& value) : buffer(std::move(value)) {}
    Buffer buffer;
  };

 public:
  struct State {
    explicit State(Mmap&& value) : loader(std::move(value)) {}
   private:
    friend class TFETSharedSegmentDataLoader;
    Mmap loader;
    std::mutex mutex;
    std::map<std::pair<size_t, size_t>, std::weak_ptr<Mapping>> mappings;
  };

  static Result<std::shared_ptr<State>> createState(const char* path) noexcept {
    if (!path || !*path) return Error::InvalidArgument;
    try {
      auto loader = Mmap::from(path, Mmap::MlockConfig::NoMlock);
      if (!loader.ok()) return loader.error();
      return std::make_shared<State>(std::move(loader.get()));
    } catch (const std::bad_alloc&) {
      return Error::MemoryAllocationFailed;
    } catch (...) {
      return Error::Internal;
    }
  }

  explicit TFETSharedSegmentDataLoader(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  Result<Buffer> load(size_t offset, size_t bytes,
                      const SegmentInfo& info) const override {
    if (!state_) return Error::InvalidState;
    try {
      std::lock_guard<std::mutex> guard(state_->mutex);
      const auto key = std::make_pair(offset, bytes);
      std::shared_ptr<Mapping> mapping;
      auto found = state_->mappings.find(key);
      if (found != state_->mappings.end()) mapping = found->second.lock();
      if (!mapping) {
        auto loaded = state_->loader.load(offset, bytes, info);
        if (!loaded.ok()) return loaded.error();
        if (bytes == 0) return std::move(loaded.get());
        mapping = std::make_shared<Mapping>(std::move(loaded.get()));
        state_->mappings[key] = mapping;
      }
      auto lease = std::make_unique<std::shared_ptr<Mapping>>(mapping);
      return Buffer(mapping->buffer.data(), mapping->buffer.size(), release,
                    lease.release());
    } catch (const std::bad_alloc&) {
      return Error::MemoryAllocationFailed;
    } catch (...) {
      return Error::Internal;
    }
  }

  Result<size_t> size() const override {
    return state_ ? state_->loader.size() : Result<size_t>(Error::InvalidState);
  }

  Error load_into(size_t offset, size_t bytes, const SegmentInfo& info,
                  void* destination) const override {
    if (!destination) return Error::InvalidArgument;
    auto loaded = load(offset, bytes, info);
    if (!loaded.ok()) return loaded.error();
    // Only this explicitly copying DataLoader API copies bytes. Module constant
    // loading uses load(), which returns the shared read-only mapped address.
    if (bytes) std::memcpy(destination, loaded->data(), bytes);
    return Error::Ok;
  }

 private:
  static void release(void* lease, void*, size_t) noexcept {
    delete static_cast<std::shared_ptr<Mapping>*>(lease);
  }
  std::shared_ptr<State> state_;
};
