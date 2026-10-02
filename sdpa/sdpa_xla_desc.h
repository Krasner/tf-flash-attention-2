#pragma once
// Descriptors passed from the XLA op kernels (sdpa_xla_kernel.cc) to the
// custom-call targets (sdpa_xla_target.cc) through the custom call's opaque
// string. Both sides include this header so the layouts can't drift apart.
//
// seed and offset are not in the descriptors: they are runtime operands (u64
// scalars in device memory), so changing them doesn't trigger a recompile.

#include <cstdint>

namespace tensorflow
{

enum FlashAttnDtype : int32_t
{
    kFlashAttnF16 = 0,
    kFlashAttnBF16 = 1,
    kFlashAttnF32 = 2,
};

// "tf_flash_attn_fwd": operands [Q, K, V, seed, offset], outputs (Out, Stats).
struct FlashAttnDesc
{
    int32_t dtype; // FlashAttnDtype
    int64_t B, Sq, Sk, Dqk, Dv;
    float dropout;
    float scale;
    bool causal;
};

// "tf_flash_attn_bwd": operands [Q, K, V, Out, Stats, dO, seed, offset],
// outputs (dQ, dK, dV, workspace). The workspace output is scratch memory for
// FlashAttnGradFunctor (custom calls can't request temporaries) and is discarded.
struct FlashAttnBwdDesc
{
    int32_t dtype; // FlashAttnDtype
    int64_t B, Sq, Sk, Dqk, Dv;
    int64_t workspace_bytes;
    float dropout;
    float scale;
    bool causal;
};

} // namespace tensorflow
