// Minimal stand-in for the PyTorch header FlashAttention-2 includes. Only
// at::PhiloxCudaState is needed: seed/offset either by value, or (captured_)
// as pointers to device memory.
#pragma once
#include <cstdint>

namespace at {
struct PhiloxCudaState {
    PhiloxCudaState() = default;
    PhiloxCudaState(uint64_t seed, uint64_t offset) {
        seed_.val = seed;
        offset_.val = offset;
    }
    PhiloxCudaState(const uint64_t *seed_dev, const uint64_t *offset_dev) {
        seed_.ptr = reinterpret_cast<int64_t *>(const_cast<uint64_t *>(seed_dev));
        offset_.ptr = reinterpret_cast<int64_t *>(const_cast<uint64_t *>(offset_dev));
        captured_ = true;
    }
    union Payload {
        uint64_t val;
        int64_t *ptr;
    };
    Payload seed_{};
    Payload offset_{};
    uint32_t offset_intragraph_ = 0;
    bool captured_ = false;
};
} // namespace at
