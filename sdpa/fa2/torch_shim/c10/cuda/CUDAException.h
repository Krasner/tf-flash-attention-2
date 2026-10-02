// Minimal stand-in for the PyTorch header FlashAttention-2 includes.
// Instead of throwing, the first error is recorded so the caller (fa2_api.cu)
// can report it through TF/XLA.
#pragma once
#include <cstdio>  // FA2 sources rely on these arriving via the real PyTorch headers.
#include <cstdlib>
#include <cuda_runtime.h>

namespace fa2_shim {
inline thread_local cudaError_t last_error = cudaSuccess;
inline void record(cudaError_t err) {
    if (err != cudaSuccess && last_error == cudaSuccess) last_error = err;
}
} // namespace fa2_shim

#define C10_CUDA_CHECK(EXPR) fa2_shim::record(EXPR)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() fa2_shim::record(cudaGetLastError())
