"""Checks the FlashAttention-2 (fp16/bf16) path of the Sdpa op, eager and under XLA,
against a float32 reference, then benchmarks it."""
import time

import numpy as np
import tensorflow as tf

mod = tf.load_op_library("./sdpa/sdpa.so")
SEED = tf.constant(42, tf.uint64)
OFFSET = tf.constant(1000, tf.uint64)


@tf.RegisterGradient("Sdpa")
def _sdpa_grad(op, d_o, *args):
    q, k, v, seed, offset = op.inputs
    o, s = op.outputs
    dq, dk, dv = mod.sdpa_grad(q, k, v, o, s, d_o, seed, offset,
                               dropout=op.get_attr("dropout"), scale=op.get_attr("scale"),
                               causal_mask=op.get_attr("causal_mask"))
    return dq, dk, dv, None, None


def flash(q, k, v, scale, causal=False, dropout=0.0):
    return mod.sdpa(q, k, v, SEED, OFFSET, dropout=dropout, scale=scale, causal_mask=causal)


def reference(q, k, v, scale, causal=False, keep_mask=None, dropout=0.0):
    q, k, v = (tf.cast(t, tf.float32) for t in (q, k, v))
    s = tf.matmul(q, k, transpose_b=True) * scale
    if causal:
        n = s.shape[-1]
        s += tf.cast(tf.range(n)[:, None] < tf.range(n)[None, :], tf.float32) * -1e9
    lse = tf.reduce_logsumexp(s, axis=-1)
    p = tf.nn.softmax(s, axis=-1)
    if keep_mask is not None:
        p = p * keep_mask / (1.0 - dropout)
    return tf.matmul(p, v), lse


def grads(f, q, k, v, do):
    with tf.GradientTape() as tape:
        tape.watch([q, k, v])
        out, lse = f(q, k, v)
        loss = tf.reduce_sum(tf.cast(out, tf.float32) * do)
    return [out, lse] + tape.gradient(loss, [q, k, v])


def max_rel_err(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    # Floor the denominator: some gradients are exactly 0 (e.g. a single key).
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-3)


def check(B, S_q, S_kv, D, causal, dropout=0.0, dtype=tf.float16, D_v=None):
    D_v = D_v or D
    tol = 5e-2 if dtype == tf.bfloat16 else 2e-2  # bf16 has 3 fewer mantissa bits
    scale = 1.0 / np.sqrt(D)
    rnd = lambda *s: tf.random.normal(s, dtype=tf.float32)
    q, k, v = rnd(B, S_q, D), rnd(B, S_kv, D), rnd(B, S_kv, D_v)
    # Round the inputs so the reference sees exactly what the kernel sees.
    q16, k16, v16 = (tf.cast(t, dtype) for t in (q, k, v))
    q, k, v = (tf.cast(t, tf.float32) for t in (q16, k16, v16))
    do = rnd(B, S_q, D_v)

    keep = None
    if dropout > 0:
        # Out is linear in V, so V = I (requires S_kv == D) returns the dropped,
        # rescaled probabilities and hence FA2's exact dropout mask.
        assert S_kv == D
        eye = tf.eye(D, batch_shape=[B], dtype=dtype)
        p_drop, _ = flash(q16, k16, eye, scale, causal, dropout)
        keep = tf.cast(p_drop > 0, tf.float32)

    flash_fn = lambda a, b, c: flash(a, b, c, scale, causal, dropout)
    got = grads(flash_fn, q16, k16, v16, do)
    want = grads(lambda a, b, c: reference(a, b, c, scale, causal, keep, dropout), q, k, v, do)
    # Forward and backward both compiled by XLA (Sdpa and SdpaGrad custom calls).
    xla_grads = tf.function(lambda a, b, c: grads(flash_fn, a, b, c, do), jit_compile=True)
    got += xla_grads(q16, k16, v16)
    want += want
    errs = [max_rel_err(g, w) for g, w in zip(got, want)]
    names = ["out", "lse", "dq", "dk", "dv"]
    names += ["xla_" + n for n in names]
    ok = all(e < tol for e in errs)
    extra = ""
    if keep is not None:
        valid = tf.ones_like(keep) if not causal else tf.linalg.band_part(tf.ones_like(keep), -1, 0)
        extra = " drop_frac=%.3f" % (1 - tf.reduce_sum(keep) / tf.reduce_sum(valid))
    dims = f"D={D}" if D_v == D else f"D_qk={D} D_v={D_v}"
    print(("OK  " if ok else "FAIL"), f"{dtype.name} B={B} Sq={S_q} Skv={S_kv} {dims} causal={causal} "
          f"dropout={dropout}: " + " ".join(f"{n}={e:.1e}" for n, e in zip(names, errs)) + extra)
    return ok


def bench(fn, n=50):
    r = fn()
    _ = r[0].numpy()
    t = time.perf_counter()
    for _ in range(n):
        r = fn()
    _ = r[0].numpy()
    return (time.perf_counter() - t) / n * 1e3


if __name__ == "__main__":
    tf.random.set_seed(0)
    all_ok = True
    for dtype in (tf.float16, tf.bfloat16):
        for D in (8, 32, 40, 64, 96, 128, 160, 192, 256):
            for causal in (False, True):
                all_ok &= check(4, 257, 257, D, causal, dtype=dtype)
        all_ok &= check(3, 100, 333, 64, False, dtype=dtype)
        all_ok &= check(2, 1, 1, 64, True, dtype=dtype)
        for causal in (False, True):
            all_ok &= check(4, 64, 64, 64, causal, dropout=0.2, dtype=dtype)
            all_ok &= check(4, 128, 128, 128, causal, dropout=0.1, dtype=dtype)
            all_ok &= check(2, 256, 256, 256, causal, dropout=0.1, dtype=dtype)
    # Shapes/dtypes FA2 doesn't handle use the op's own kernels.
    all_ok &= check(2, 100, 120, 64, False, dtype=tf.bfloat16, D_v=32)
    all_ok &= check(2, 100, 120, 64, False, dtype=tf.float32)
    all_ok &= check(2, 100, 100, 32, True, dtype=tf.float32, D_v=16)
    # Feature sizes over 128 are rejected cleanly where FA2 can't run them.
    for dtype, D, D_v in ((tf.float32, 256, 256), (tf.float16, 256, 128), (tf.float16, 264, 264)):
        q = tf.zeros((1, 16, D), dtype)
        v = tf.zeros((1, 16, D_v), dtype)
        for name, fn in (("eager", flash), ("XLA", tf.function(flash, jit_compile=True))):
            try:
                fn(q, q, v, 1.0)
                ok = False
            except (tf.errors.InvalidArgumentError, ValueError):
                ok = True
            print(("OK  " if ok else "FAIL"), f"{dtype.name} D_qk={D} D_v={D_v} {name} rejected")
            all_ok &= ok
    print("ALL OK" if all_ok else "SOME CHECKS FAILED")

    B, S_q, S_kv, D = 16, 1024, 4096, 128
    scale = 1.0 / np.sqrt(D)
    do = tf.random.normal((B, S_q, D), dtype=tf.float32)
    for dtype, dropout in ((tf.float16, 0.0), (tf.float16, 0.2), (tf.bfloat16, 0.0),
                           (tf.bfloat16, 0.2)):
        q = tf.random.normal((B, S_q, D), dtype=dtype)
        k = tf.random.normal((B, S_kv, D), dtype=dtype)
        v = tf.random.normal((B, S_kv, D), dtype=dtype)
        fwd = lambda: flash(q, k, v, scale, False, dropout)
        fwd_xla = tf.function(fwd, jit_compile=True)

        fwd_bwd_fn = lambda: grads(lambda a, b, c: flash(a, b, c, scale, False, dropout),
                                   q, k, v, do)[2:]
        fwd_bwd = tf.function(fwd_bwd_fn)
        fwd_bwd_xla = tf.function(fwd_bwd_fn, jit_compile=True)

        print(f"B={B} Sq={S_q} Skv={S_kv} D={D} {dtype.name} dropout={dropout}: "
              f"fwd {bench(fwd):.2f} ms, fwd XLA {bench(fwd_xla):.2f} ms, "
              f"fwd+bwd {bench(fwd_bwd):.2f} ms, fwd+bwd XLA {bench(fwd_bwd_xla):.2f} ms")
