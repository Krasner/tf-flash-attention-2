# Sourced by build.sh and fa2/build_fa2.sh. Decides which GPU architectures to
# compile for and exports:
#   CUDA_ARCHS     all target archs, e.g. "80 86"
#   FA2_ARCHS      the subset FlashAttention-2 supports (sm80+); may be empty
#   gencode_flags  function: gencode_flags 80 86 -> nvcc -gencode flags
#
# By default the archs are those of the GPUs visible on the build machine
# (via nvidia-smi). Override with CUDA_ARCHS, e.g. for a build host without
# a GPU or to target several GPU types:
#   CUDA_ARCHS="80 86 90" ./build.sh      (also accepts "8.0;8.6;9.0")

cudapath="${cudapath:-/usr/local/cuda}"

if [ -z "${CUDA_ARCHS:-}" ]; then
    if ! command -v nvidia-smi > /dev/null || \
        ! CUDA_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null) || \
        [ -z "$CUDA_ARCHS" ]; then
        echo "error: could not query GPU compute capability with nvidia-smi." >&2
        echo "Set it explicitly, e.g. CUDA_ARCHS=\"80\" ./build.sh" >&2
        exit 1
    fi
fi

# Normalise "8.0;8.6,9.0 9.0" -> "80 86 90" (sorted, unique).
CUDA_ARCHS=$(echo "$CUDA_ARCHS" | tr ';, ' '\n\n\n' | sed 's/\.//; /^$/d' | sort -n -u | tr '\n' ' ')
CUDA_ARCHS="${CUDA_ARCHS% }"

supported_archs=" $("$cudapath/bin/nvcc" --list-gpu-arch | sed 's/compute_//' | tr '\n' ' ') "
FA2_ARCHS=""
for arch in $CUDA_ARCHS; do
    if [[ ! "$arch" =~ ^[0-9]+$ ]] || [[ "$supported_archs" != *" $arch "* ]]; then
        echo "error: sm_$arch is not supported by $cudapath/bin/nvcc (supports:${supported_archs% })" >&2
        exit 1
    fi
    if [ "$arch" -ge 80 ]; then
        FA2_ARCHS="$FA2_ARCHS $arch"
    fi
done
FA2_ARCHS="${FA2_ARCHS# }"

# SASS for every arch, plus PTX for the newest one so the binary can still be
# JIT-compiled on GPUs newer than any listed.
gencode_flags() {
    local arch flags="" newest=""
    for arch in "$@"; do
        flags="$flags -gencode arch=compute_${arch},code=sm_${arch}"
        newest="$arch"
    done
    [ -n "$newest" ] && flags="$flags -gencode arch=compute_${newest},code=compute_${newest}"
    echo "${flags# }"
}

export CUDA_ARCHS FA2_ARCHS
