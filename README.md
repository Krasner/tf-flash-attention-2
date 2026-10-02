# Flash Attention 2 - TensorFlow Custom Op

Closely follows this custom tensorflow implementation:
https://github.com/gravytrain777/tf-custom-flash-attention

Wraps https://github.com/Dao-AILab/flash-attention for use in tensorflow. Allows XLA.

Most code developed with Claude.

## Features

- CPU and GPU implementations
- Causal masking support
- Dropout support

## Installation

Install third party dependencies:
```
sudo apt-get update
sudo apt-get install llvm-dev

git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git third_party/flash-attention

git -C third_party/flash-attention submodule update --init --depth 1 csrc/cutlass
```

### Build from Source
```
export TF_PATH=/path/to/python/site-packages/tensorflow
export CUDA_PATH=/usr/local/cuda
./build.sh
```

Test build with `python test_fa2.py`

## Usage

### 1. Register the Custom Operation

Create a Python script to load and register the gradient operation:

```python
import tensorflow as tf

# Load the custom op
mod = tf.load_op_library("./sdpa/sdpa.so")

def sdpa(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
         seed: tf.Tensor, offset: tf.Tensor, 
         dropout: float = 0.0, 
         scale: float = 1.0, 
         causal_mask: bool = False):
    o, s = mod.sdpa(q, k, v, seed, offset,
                    dropout=dropout, scale=scale, causal_mask=causal_mask)
    return o, s

@tf.RegisterGradient("Sdpa")
def _sdpa_grad(op: tf.Operation, d_o, *args):
    q, k, v, seed, offset = op.inputs
    o, s = op.outputs
    dropout = op.get_attr("dropout")
    scale = op.get_attr("scale")
    causal_mask = op.get_attr("causal_mask")
    
    dq, dk, dv = mod.sdpa_grad(
        q, k, v, o, s, d_o, seed, offset,
        dropout=dropout, scale=scale, causal_mask=causal_mask
    )
    return dq, dk, dv, None, None
```

### 2. Example

```python
import numpy as np

# Configuration
B = 20          # Batch size
S_q = 400       # Query sequence length
S_kv = 400      # Key/Value sequence length
D_qk = 32       # Query/Key dimension
D_v = 4         # Value dimension
dropout = 0.2
scale = 1.0 / np.sqrt(D_qk)
causal_mask = False

# Random seed for reproducibility (important to use uint64)
seed = tf.constant(42, dtype=tf.uint64)
offset = tf.constant(1000, dtype=tf.uint64)

# Input tensors
q = tf.random.normal(shape=(B, S_q, D_qk), dtype=tf.bfloat16)
k = tf.random.normal(shape=(B, S_kv, D_qk), dtype=tf.bfloat16)
v = tf.random.normal(shape=(B, S_kv, D_v), dtype=tf.bfloat16)

# GPU execution with gradient computation
with tf.device('/GPU:0'):
    with tf.GradientTape() as tape:
        tape.watch([q, k, v])
        out, _ = sdpa(q=q, k=k, v=v, 
                     seed=seed, offset=offset,
                     dropout=dropout, 
                     scale=scale, 
                     causal_mask=causal_mask)
        loss = tf.reduce_sum(tf.reduce_max(out, axis=-1))
    
    gradients = tape.gradient(loss, [q, k, v])

print("Output:", out.numpy())
print("Gradients dQ:", gradients[0].numpy())
```

## Details

### Input Requirements
- All inputs must be **3D tensors** with shape `[batch, sequence, dimension]`
- Supported dtypes: `float16`, `bfloat16`

### CPU vs GPU
- **CPU**: Uses memory-efficient attention not strictly Flash Attention
- **GPU**: Implements Flash Attention with self-optimized CUDA kernels

### Backward Pass
The GPU backward pass uses two separate kernels:
1. Compute dQ 
2. Compute dK & dV 

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention Repository](https://github.com/Dao-AILab/flash-attention)
- [tf custom flash attention](https://github.com/gravytrain777/tf-custom-flash-attention)
