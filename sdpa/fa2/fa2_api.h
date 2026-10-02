#pragma once
// Thin wrapper around the vendored FlashAttention-2 kernels. Kept free of
// TensorFlow and PyTorch headers so it can be compiled with FA2's own flags.
//
// All tensors are contiguous and row-major: q/out are [B, S_q, D], k/v are
// [B, S_kv, D] (fp16, or bf16 when bf16 = true), lse is float [B, S_q].
// Heads are expected to be folded into B, matching the Sdpa op.
//
// Dropout RNG: seed/offset are used by value unless seed_dev/offset_dev are
// non-null, in which case they point to device memory holding the values
// (read by the kernels, so no host sync is needed; used by the XLA path).

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace sdpa_fa2
{

// True if the current device and shapes can use FlashAttention-2: the library
// was built with FA2 code for this GPU (sm80+), D_qk == D_v, and the head dim
// is a multiple of 8 and <= 128.
bool Supported(int d_qk, int d_v);

// The shape part of Supported: true if FA2 could handle these head dims on a
// suitable GPU. Doesn't touch the device, so it's safe at XLA compile time.
bool ShapeSupported(int d_qk, int d_v);

cudaError_t Forward(cudaStream_t stream, bool bf16, const void *q, const void *k, const void *v,
                    void *out, float *lse, int B, int S_q, int S_kv, int D, float scale,
                    float dropout, bool causal, uint64_t seed, uint64_t offset,
                    const uint64_t *seed_dev, const uint64_t *offset_dev);

// Scratch memory Backward needs (dQ accumulator, softmax row sums, RNG state).
size_t BackwardWorkspaceBytes(int B, int S_q, int D);

cudaError_t Backward(cudaStream_t stream, bool bf16, const void *q, const void *k,
                     const void *v, const void *out, const float *lse, const void *dout, void *dq,
                     void *dk, void *dv, void *workspace, int B, int S_q, int S_kv, int D, float scale,
                     float dropout, bool causal, uint64_t seed, uint64_t offset,
                     const uint64_t *seed_dev, const uint64_t *offset_dev);

} // namespace sdpa_fa2
