export TF_PATH="${TF_PATH}"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export LD_LIBRARY_PATH=${CUDA_PATH}:/usr/local/isl/lib/:$LD_LIBRARY_PATH

if [[ -z "$TF_PATH" ]]; then
    echo "Assign tensorflow path TF_PATH=</path/to/python/site-packages/tensorflow>"
    exit 1
fi

set -e

cd ./sdpa

cudapath=${CUDA_PATH}
tfpath=${TF_PATH}

# Target GPU archs: detected from the GPUs on this machine, or set CUDA_ARCHS
# (e.g. CUDA_ARCHS="80 86 90" ./build.sh). See cuda_arch.sh.
source ./cuda_arch.sh
CUDA_ARCH=$(gencode_flags $CUDA_ARCHS)
echo "Building for CUDA archs: $CUDA_ARCHS"

# FlashAttention-2 kernels used for fp16/bf16 on sm80+ (slow to compile; cached in fa2/obj).
./fa2/build_fa2.sh
FA2_OBJS=$(cat fa2/obj/objects.txt)

OP_NAME="sdpa"
sosuffix="so"

soname="${OP_NAME}.${sosuffix}"

OP_SOURCE="${OP_NAME}_ops.cc"
OP_OUT="${OP_NAME}_ops.o"

CPU_SOURCE="${OP_NAME}_cpu.cc"
CPU_OUT="${OP_NAME}_cpu.o"

GPU_FWD_SOURCE="${OP_NAME}_fwd.cu.cc"
GPU_FWD_OUT="${OP_NAME}_fwd.cu.o"

GPU_BWD_SOURCE="${OP_NAME}_bwd.cu.cc"
GPU_BWD_OUT="${OP_NAME}_bwd.cu.o"

XLA_TARGET_SOURCE="${OP_NAME}_xla_target.cc"
XLA_KERNEL_SOURCE="${OP_NAME}_xla_kernel.cc"
XLA_KERNEL_OUT="${OP_NAME}_xla_kernel.o"
XLA_TARGET_OUT="${OP_NAME}_xla_target.o"

TF_CFLAGS="-I$tfpath/include -D_GLIBCXX_USE_CXX11_ABI=1 --std=c++17 -DEIGEN_MAX_ALIGN_BYTES=64"
TF_LFLAGS="-L$tfpath -l:libtensorflow_framework.so.2 -l:libtensorflow_cc.so.2"

cuda_lib_path="$cudapath/lib64"
cudart_lib="cudart"
# CUDA_LINK="-Wl,-rpath,${cuda_lib_path} -L${cuda_lib_path} -l${cudart_lib}"
CUDA_LINK="-L${cuda_lib_path} -l${cudart_lib}"

$cudapath/bin/nvcc -c ${GPU_FWD_SOURCE} -o ${GPU_FWD_OUT} \
    $TF_CFLAGS -D GOOGLE_CUDA=1 -x cu \
    -ccbin=/usr/bin/g++ -Xcompiler "-fPIC -Wno-deprecated-declarations -Wno-attributes -O2"\
    --expt-relaxed-constexpr -diag-suppress 2810,611 -O2 $CUDA_ARCH

$cudapath/bin/nvcc -c ${GPU_BWD_SOURCE} -o ${GPU_BWD_OUT} \
    $TF_CFLAGS -D GOOGLE_CUDA=1 -x cu \
    -ccbin=/usr/bin/g++ -Xcompiler "-fPIC -Wno-deprecated-declarations -Wno-attributes -O2"\
    --expt-relaxed-constexpr -diag-suppress 2810,611 -O2 $CUDA_ARCH

/usr/bin/g++ -c $OP_SOURCE -o $OP_OUT -fPIC $TF_CFLAGS -D GOOGLE_CUDA=1 -O2

/usr/bin/g++ -c $CPU_SOURCE -o $CPU_OUT -fPIC $TF_CFLAGS -D GOOGLE_CUDA=1 -O2

/usr/bin/g++ -c $XLA_KERNEL_SOURCE -o $XLA_KERNEL_OUT -fPIC $TF_CFLAGS -D GOOGLE_CUDA=1 -DEIGEN_USE_GPU -O2

/usr/bin/g++ -c $XLA_TARGET_SOURCE -o $XLA_TARGET_OUT -fPIC $TF_CFLAGS -D GOOGLE_CUDA=1 -DEIGEN_USE_GPU -O2

/usr/bin/g++ -std=c++17 -shared -o $soname $OP_OUT $CPU_OUT $GPU_FWD_OUT $GPU_BWD_OUT $XLA_KERNEL_OUT $XLA_TARGET_OUT $FA2_OBJS \
    -fPIC ${TF_LFLAGS} ${CUDA_LINK} -mavx2 -O2 \
    -march=native -mtune=native \
     -D GOOGLE_CUDA=1 
