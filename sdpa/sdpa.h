#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "unsupported/Eigen/CXX11/Tensor"

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace tensorflow::functor
{
// Dropout RNG seed and offset. Normally given by value; the XLA path only has
// them in device memory, in which case seed_dev/offset_dev point there and the
// GPU kernels read them on the device (seed/offset are then ignored).
struct SdpaRng
{
    uint64 seed = 0;
    uint64 offset = 0;
    const uint64 *seed_dev = nullptr;
    const uint64 *offset_dev = nullptr;
};

// Error for feature sizes the kernels can't handle. Up to 128 always works on GPU;
// up to 256 only on the FlashAttention-2 path (see fa2/fa2_api.h).
inline Status FeatureSizeError(int D_qk, int D_v)
{
    return errors::InvalidArgument(
        "Feature size D_qk=", D_qk, ", D_v=", D_v,
        " is not supported. The GPU kernels support up to 128; up to 256 requires "
        "float16/bfloat16 on an sm80+ GPU with D_qk == D_v and a multiple of 8 (and, for "
        "D > 192 with dropout, >= 144 KB shared memory per block, e.g. A100/H100). Set smaller "
        "feature size or use more heads");
}

template <typename Device, typename T> struct FlashAttnFunctor
{
    // OK if the kernels for this device and dtype handle these feature sizes.
    static Status CheckFeatureSize(int D_qk, int D_v);

    void operator()(const Device &d,  typename TTypes<T, 3>::ConstTensor Q,
                    typename TTypes<T, 3>::ConstTensor K, typename TTypes<T, 3>::ConstTensor V,
                    typename TTypes<T, 3>::Tensor Out, typename TTypes<float, 2>::Tensor stats, 
                    bool causal_mask, float dropout_rate, float scale, SdpaRng rng)
                     const;
};

template <typename Device, typename T>
struct FlashAttnGradFunctor
{
    // Bytes of device scratch memory operator() needs as `workspace` (may be 0).
    // Depends only on shapes, never on the current device.
    static int64 WorkspaceBytes(int B, int S_q, int S_kv, int D_qk, int D_v);

    // OK if the kernels for this device and dtype handle these feature sizes.
    static Status CheckFeatureSize(int D_qk, int D_v, float dropout_rate);

    void operator()(const Device &d,
                    typename TTypes<T, 3>::ConstTensor Q,
                    typename TTypes<T, 3>::ConstTensor K,
                    typename TTypes<T, 3>::ConstTensor V,
                    typename TTypes<T, 3>::ConstTensor Out,
                    typename TTypes<float, 2>::ConstTensor Stats,
                    typename TTypes<T, 3>::ConstTensor dO,
                    typename TTypes<T, 3>::Tensor dQ,
                    typename TTypes<T, 3>::Tensor dK,
                    typename TTypes<T, 3>::Tensor dV,
                    bool causal_mask, float dropout_rate, float scale,
                    SdpaRng rng, void *workspace) const;
};

}; // namespace tensorflow::functor