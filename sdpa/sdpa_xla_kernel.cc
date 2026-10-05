// XLA lowerings of Sdpa / SdpaGrad to custom calls (targets in sdpa_xla_target.cc).
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/hlo/builder/xla_builder.h"  // path varies by TF version
#include "tensorflow/compiler/xla/shape_util.h"

#include "fa2/fa2_api.h"
#include "sdpa.h"
#include "sdpa_xla_desc.h"

namespace tensorflow {
namespace {

// seed/offset must be uint64 scalars (they become u64[] custom-call operands).
static Status ValidateRngInputs(XlaOpKernelContext* ctx) {
  for (absl::string_view name : {"seed", "offset"}) {
    const TensorShape shape = ctx->InputShape(name);
    if (!TensorShapeUtils::IsScalar(shape)) {
      return errors::InvalidArgument(name, " must be a scalar, got ", shape.DebugString());
    }
  }
  return OkStatus();
}

static Status ToFlashAttnDtype(DataType dt, int32_t* out) {
  switch (dt) {
    case DT_HALF:     *out = kFlashAttnF16; return OkStatus();
    case DT_BFLOAT16: *out = kFlashAttnBF16; return OkStatus();
    case DT_FLOAT:    *out = kFlashAttnF32; return OkStatus();
    default:          return errors::InvalidArgument("unsupported dtype ", DataTypeString(dt));
  }
}

// Same restrictions as the TF kernels in sdpa_cpu.cc, except that feature sizes
// over 128 are checked by shape only (no device query at compile time): if the
// device turns out not to support them, the custom call fails at run time.
static Status ValidateQKV(const TensorShape& q, const TensorShape& k, const TensorShape& v,
                          bool causal, DataType dt) {
  if (q.dims() != 3 || k.dims() != 3 || v.dims() != 3) {
    return errors::InvalidArgument("Q/K/V must be rank 3");
  }
  if (k.dim_size(0) != q.dim_size(0) || v.dim_size(0) != q.dim_size(0)) {
    return errors::InvalidArgument("Q/K/V batch size mismatch");
  }
  if (k.dim_size(2) != q.dim_size(2)) {
    return errors::InvalidArgument("K feature size mismatch");
  }
  if (v.dim_size(1) != k.dim_size(1)) {
    return errors::InvalidArgument("V seq size mismatch");
  }
  if (causal && q.dim_size(1) != k.dim_size(1)) {
    return errors::InvalidArgument("seq size Q != K while causal_mask = True");
  }
  const int64_t d_qk = q.dim_size(2), d_v = v.dim_size(2);
  const bool fa2_type = dt == DT_HALF || dt == DT_BFLOAT16;
  if (std::max(d_qk, d_v) > 128 && !(fa2_type && sdpa_fa2::ShapeSupported(d_qk, d_v))) {
    return functor::FeatureSizeError(d_qk, d_v);
  }
  return OkStatus();
}

static Status ExpectShape(absl::string_view name, const TensorShape& got,
                          const TensorShape& want) {
  if (got != want) {
    return errors::InvalidArgument(name, " has shape ", got.DebugString(), ", expected ",
                                   want.DebugString());
  }
  return OkStatus();
}

// Dense row-major layout, matching Eigen RowMajor TensorMaps.
static xla::Shape RowMajor(xla::PrimitiveType t, std::initializer_list<int64_t> dims) {
  return xla::ShapeUtil::MakeShapeWithDescendingLayout(t, dims);
}

template <typename Desc>
static std::string ToOpaque(const Desc& d) {
  return std::string(reinterpret_cast<const char*>(&d), sizeof(d));
}

class FlashAttnXlaOp : public XlaOpKernel {
 public:
  explicit FlashAttnXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("causal_mask", &causal_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout", &dropout_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &scale_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    FlashAttnDesc d{};
    const TensorShape q = ctx->InputShape(0);
    const TensorShape k = ctx->InputShape(1);
    const TensorShape v = ctx->InputShape(2);
    OP_REQUIRES_OK(ctx, ValidateQKV(q, k, v, causal_, ctx->input_type(0)));
    OP_REQUIRES_OK(ctx, ValidateRngInputs(ctx));
    OP_REQUIRES_OK(ctx, ToFlashAttnDtype(ctx->input_type(0), &d.dtype));
    xla::PrimitiveType ptype;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->input_type(0), &ptype));

    d.B = q.dim_size(0); d.Sq = q.dim_size(1); d.Dqk = q.dim_size(2);
    d.Sk = k.dim_size(1); d.Dv = v.dim_size(2);
    d.dropout = dropout_; d.scale = scale_;
    d.causal = causal_;

    xla::Shape u64_s = RowMajor(xla::U64, {});
    xla::Shape q_s   = RowMajor(ptype, {d.B, d.Sq, d.Dqk});
    xla::Shape k_s   = RowMajor(ptype, {d.B, d.Sk, d.Dqk});
    xla::Shape v_s   = RowMajor(ptype, {d.B, d.Sk, d.Dv});
    xla::Shape out_s = RowMajor(ptype, {d.B, d.Sq, d.Dv});
    xla::Shape st_s  = RowMajor(xla::F32, {d.B, d.Sq});

    xla::XlaOp call = xla::CustomCallWithLayout(
        ctx->builder(), "tf_flash_attn_fwd",
        /*operands=*/{ctx->Input(0), ctx->Input(1), ctx->Input(2), ctx->Input("seed"),
                      ctx->Input("offset")},
        /*shape_with_layout=*/xla::ShapeUtil::MakeTupleShape({out_s, st_s}),
        /*operand_shapes_with_layout=*/{q_s, k_s, v_s, u64_s, u64_s},
        /*opaque=*/ToOpaque(d),
        /*has_side_effect=*/false,
        /*output_operand_aliasing=*/{},
        /*literal=*/nullptr,
        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
        /*api_version=*/xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING);

    ctx->SetOutput(0, xla::GetTupleElement(call, 0));  // Out
    ctx->SetOutput(1, xla::GetTupleElement(call, 1));  // stats
  }

 private:
  bool causal_;
  float dropout_, scale_;
};

class FlashAttnGradXlaOp : public XlaOpKernel {
 public:
  explicit FlashAttnGradXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("causal_mask", &causal_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout", &dropout_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &scale_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    FlashAttnBwdDesc d{};
    const TensorShape q = ctx->InputShape(0);
    const TensorShape k = ctx->InputShape(1);
    const TensorShape v = ctx->InputShape(2);
    OP_REQUIRES_OK(ctx, ValidateQKV(q, k, v, causal_, ctx->input_type(0)));
    OP_REQUIRES_OK(ctx, ValidateRngInputs(ctx));
    OP_REQUIRES_OK(ctx, ToFlashAttnDtype(ctx->input_type(0), &d.dtype));
    xla::PrimitiveType ptype;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->input_type(0), &ptype));

    d.B = q.dim_size(0); d.Sq = q.dim_size(1); d.Dqk = q.dim_size(2);
    d.Sk = k.dim_size(1); d.Dv = v.dim_size(2);
    d.dropout = dropout_; d.scale = scale_;
    d.causal = causal_;

    const TensorShape out_shape({d.B, d.Sq, d.Dv});
    OP_REQUIRES_OK(ctx, ExpectShape("out", ctx->InputShape(3), out_shape));
    OP_REQUIRES_OK(ctx, ExpectShape("stats", ctx->InputShape(4), TensorShape({d.B, d.Sq})));
    OP_REQUIRES_OK(ctx, ExpectShape("do", ctx->InputShape(5), out_shape));

    // Sized from shapes alone, so it doesn't matter which device is current here.
    switch (d.dtype) {
      case kFlashAttnF16:
        d.workspace_bytes = functor::FlashAttnGradFunctor<GPUDevice, Eigen::half>::WorkspaceBytes(
            d.B, d.Sq, d.Sk, d.Dqk, d.Dv);
        break;
      case kFlashAttnBF16:
        d.workspace_bytes =
            functor::FlashAttnGradFunctor<GPUDevice, Eigen::bfloat16>::WorkspaceBytes(
                d.B, d.Sq, d.Sk, d.Dqk, d.Dv);
        break;
      default:
        d.workspace_bytes = functor::FlashAttnGradFunctor<GPUDevice, float>::WorkspaceBytes(
            d.B, d.Sq, d.Sk, d.Dqk, d.Dv);
    }

    xla::Shape u64_s = RowMajor(xla::U64, {});
    xla::Shape q_s  = RowMajor(ptype, {d.B, d.Sq, d.Dqk});
    xla::Shape k_s  = RowMajor(ptype, {d.B, d.Sk, d.Dqk});
    xla::Shape v_s  = RowMajor(ptype, {d.B, d.Sk, d.Dv});
    xla::Shape o_s  = RowMajor(ptype, {d.B, d.Sq, d.Dv});
    xla::Shape st_s = RowMajor(xla::F32, {d.B, d.Sq});
    xla::Shape ws_s = RowMajor(xla::U8, {d.workspace_bytes});

    xla::XlaOp call = xla::CustomCallWithLayout(
        ctx->builder(), "tf_flash_attn_bwd",
        /*operands=*/{ctx->Input(0), ctx->Input(1), ctx->Input(2), ctx->Input(3),
                      ctx->Input(4), ctx->Input(5), ctx->Input("seed"), ctx->Input("offset")},
        /*shape_with_layout=*/xla::ShapeUtil::MakeTupleShape({q_s, k_s, v_s, ws_s}),
        /*operand_shapes_with_layout=*/{q_s, k_s, v_s, o_s, st_s, o_s, u64_s, u64_s},
        /*opaque=*/ToOpaque(d),
        /*has_side_effect=*/false,
        /*output_operand_aliasing=*/{},
        /*literal=*/nullptr,
        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
        /*api_version=*/xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING);

    ctx->SetOutput(0, xla::GetTupleElement(call, 0));  // dQ
    ctx->SetOutput(1, xla::GetTupleElement(call, 1));  // dK
    ctx->SetOutput(2, xla::GetTupleElement(call, 2));  // dV
  }

 private:
  bool causal_;
  float dropout_, scale_;
};

REGISTER_XLA_OP(Name("Sdpa")
                    .Device(DEVICE_GPU_XLA_JIT)
                    .TypeConstraint("T", {DT_HALF, DT_BFLOAT16, DT_FLOAT}),
                FlashAttnXlaOp);

REGISTER_XLA_OP(Name("SdpaGrad")
                    .Device(DEVICE_GPU_XLA_JIT)
                    .TypeConstraint("T", {DT_HALF, DT_BFLOAT16, DT_FLOAT}),
                FlashAttnGradXlaOp);

}  // namespace
}  // namespace tensorflow
