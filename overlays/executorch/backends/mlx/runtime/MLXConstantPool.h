// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
#pragma once

#include "MLXConstantPoolCore.h"
#include <mlx/array.h>
#include <mlx/device.h>
#include <executorch/runtime/core/freeable_buffer.h>

namespace executorch::backends::mlx {

class MLXConstantPool final : public BasicMLXConstantPool<::mlx::core::array> {
  static const void* identity(const ::mlx::core::array& value) {
    // Source-constructed/reused values are normally already available. Never
    // evaluate or wait merely to observe an identity.
    return value.is_available() ? value.buffer().ptr() : nullptr;
  }
 public:
  explicit MLXConstantPool(MLXConstantPoolMode mode)
      : BasicMLXConstantPool(mode, identity) {}
  ~MLXConstantPool() noexcept { write_receipt(); }
};
using MLXConstantPoolLoadScope = ScopedMLXConstantPool<MLXConstantPool>;
inline std::shared_ptr<MLXConstantPool> current_mlx_constant_pool() noexcept {
  return current_mlx_constant_pool_for_type<MLXConstantPool>();
}

inline MLXConstantPoolKey make_mlx_constant_pool_key(
    const std::string& name, const runtime::FreeableBuffer& buffer,
    const ::mlx::core::Shape& shape, ::mlx::core::Dtype dtype,
    ::mlx::core::Device device) {
  size_t bytes = dtype.size();
  for (const auto dimension : shape) {
    if (dimension < 0 || (dimension && bytes > std::numeric_limits<size_t>::max() / size_t(dimension)))
      throw std::invalid_argument("Constant pool logical size overflow or invalid shape");
    bytes *= size_t(dimension);
  }
  return {buffer.data(), buffer.size(), bytes, name,
          std::vector<int>(shape.begin(), shape.end()), int(dtype.val()), int(device.type), device.index};
}

inline ::mlx::core::array load_mlx_pooled_constant(
    MLXConstantPool::Load& load, const MLXConstantPoolKey& key,
    runtime::FreeableBuffer&& source, const ::mlx::core::Shape& shape,
    ::mlx::core::Dtype dtype) {
  auto lease = std::make_shared<runtime::FreeableBuffer>(std::move(source));
  return load.acquire(key, lease, [&](const std::shared_ptr<void>& owner) {
    // Each returned array retains the source itself. This remains safe if a
    // handle, pool or load scope is destroyed before a downstream graph copy.
    return ::mlx::core::array(const_cast<void*>(key.source), shape, dtype,
        [owner](void*) {});
  });
}

} // namespace executorch::backends::mlx
