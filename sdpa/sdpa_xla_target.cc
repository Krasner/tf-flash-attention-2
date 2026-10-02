// XLA custom-call targets for Sdpa / SdpaGrad (lowered in sdpa_xla_kernel.cc).
// They wrap XLA's stream in an Eigen device and call the same functors as the
// TF kernels.
#define EIGEN_USE_GPU
#include <cuda_runtime.h>
#include <cstring>
#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/core/platform/logging.h"
#include "sdpa.h"
#include "sdpa_xla_desc.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <typename T>
void RunFlashAttn(cudaStream_t stream, void** buffers, const FlashAttnDesc& d) {
  Eigen::GpuStreamDevice stream_device(&stream, /*device=*/-1);  // -1 = current device
  GPUDevice device(&stream_device);

  // buffers = [Q, K, V, seed, offset, Out, stats]  (operands, then tuple outputs flattened)
  typename TTypes<T, 3>::ConstTensor Q(static_cast<const T*>(buffers[0]), d.B, d.Sq, d.Dqk);
  typename TTypes<T, 3>::ConstTensor K(static_cast<const T*>(buffers[1]), d.B, d.Sk, d.Dqk);
  typename TTypes<T, 3>::ConstTensor V(static_cast<const T*>(buffers[2]), d.B, d.Sk, d.Dv);
  functor::SdpaRng rng;
  rng.seed_dev = static_cast<const uint64*>(buffers[3]);
  rng.offset_dev = static_cast<const uint64*>(buffers[4]);
  typename TTypes<T, 3>::Tensor Out(static_cast<T*>(buffers[5]), d.B, d.Sq, d.Dv);
  typename TTypes<float, 2>::Tensor stats(static_cast<float*>(buffers[6]), d.B, d.Sq);

  functor::FlashAttnFunctor<GPUDevice, T>()(device, Q, K, V, Out, stats, d.causal, d.dropout,
                                            d.scale, rng);
}

template <typename T>
void RunFlashAttnGrad(cudaStream_t stream, void** buffers, const FlashAttnBwdDesc& d) {
  Eigen::GpuStreamDevice stream_device(&stream, /*device=*/-1);  // -1 = current device
  GPUDevice device(&stream_device);

  // buffers = [Q, K, V, Out, stats, dO, seed, offset, dQ, dK, dV, workspace]
  typename TTypes<T, 3>::ConstTensor Q(static_cast<const T*>(buffers[0]), d.B, d.Sq, d.Dqk);
  typename TTypes<T, 3>::ConstTensor K(static_cast<const T*>(buffers[1]), d.B, d.Sk, d.Dqk);
  typename TTypes<T, 3>::ConstTensor V(static_cast<const T*>(buffers[2]), d.B, d.Sk, d.Dv);
  typename TTypes<T, 3>::ConstTensor Out(static_cast<const T*>(buffers[3]), d.B, d.Sq, d.Dv);
  typename TTypes<float, 2>::ConstTensor stats(static_cast<const float*>(buffers[4]), d.B,
                                               d.Sq);
  typename TTypes<T, 3>::ConstTensor dO(static_cast<const T*>(buffers[5]), d.B, d.Sq, d.Dv);
  functor::SdpaRng rng;
  rng.seed_dev = static_cast<const uint64*>(buffers[6]);
  rng.offset_dev = static_cast<const uint64*>(buffers[7]);
  typename TTypes<T, 3>::Tensor dQ(static_cast<T*>(buffers[8]), d.B, d.Sq, d.Dqk);
  typename TTypes<T, 3>::Tensor dK(static_cast<T*>(buffers[9]), d.B, d.Sk, d.Dqk);
  typename TTypes<T, 3>::Tensor dV(static_cast<T*>(buffers[10]), d.B, d.Sk, d.Dv);
  void* workspace = d.workspace_bytes > 0 ? buffers[11] : nullptr;

  functor::FlashAttnGradFunctor<GPUDevice, T>()(device, Q, K, V, Out, stats, dO, dQ, dK, dV,
                                                d.causal, d.dropout, d.scale, rng, workspace);
}

static void LogLaunchError(const char* what) {
  if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
    LOG(ERROR) << what << " launch failed: " << cudaGetErrorString(err);
  }
}

void FlashAttnFwdCustomCall(cudaStream_t stream, void** buffers,
                            const char* opaque, size_t opaque_len,
                            XlaCustomCallStatus* status) {
  if (opaque_len != sizeof(FlashAttnDesc)) {
    LOG(FATAL) << "FlashAttn: bad descriptor size " << opaque_len;
    return;
  }
  FlashAttnDesc d;
  std::memcpy(&d, opaque, sizeof(d));

  switch (d.dtype) {
    case kFlashAttnF16: RunFlashAttn<Eigen::half>(stream, buffers, d); break;
    case kFlashAttnBF16: RunFlashAttn<Eigen::bfloat16>(stream, buffers, d); break;
    case kFlashAttnF32: RunFlashAttn<float>(stream, buffers, d); break;
    default: LOG(FATAL) << "FlashAttn: unsupported dtype " << d.dtype;
  }
  LogLaunchError("FlashAttn");
}

void FlashAttnBwdCustomCall(cudaStream_t stream, void** buffers,
                            const char* opaque, size_t opaque_len,
                            XlaCustomCallStatus* status) {
  if (opaque_len != sizeof(FlashAttnBwdDesc)) {
    LOG(FATAL) << "FlashAttnGrad: bad descriptor size " << opaque_len;
    return;
  }
  FlashAttnBwdDesc d;
  std::memcpy(&d, opaque, sizeof(d));

  switch (d.dtype) {
    case kFlashAttnF16: RunFlashAttnGrad<Eigen::half>(stream, buffers, d); break;
    case kFlashAttnBF16: RunFlashAttnGrad<Eigen::bfloat16>(stream, buffers, d); break;
    case kFlashAttnF32: RunFlashAttnGrad<float>(stream, buffers, d); break;
    default: LOG(FATAL) << "FlashAttnGrad: unsupported dtype " << d.dtype;
  }
  LogLaunchError("FlashAttnGrad");
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("tf_flash_attn_fwd",
                                         FlashAttnFwdCustomCall, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("tf_flash_attn_bwd",
                                         FlashAttnBwdCustomCall, "CUDA");

}  // namespace tensorflow
