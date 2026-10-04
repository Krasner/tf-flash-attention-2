#!/bin/bash
# Builds the FlashAttention-2 kernels (vendored in third_party/flash-attention)
# plus the thin C++ glue in fa2_api.cu. No TensorFlow or PyTorch headers are
# involved; the few PyTorch headers FA2 includes are replaced by torch_shim/.
#
# Target archs come from ../cuda_arch.sh (FA2_ARCHS). If none of them is sm80+,
# only a stub fa2_api is built and the op falls back to its own CUDA kernels.
#
# Objects are cached per arch set in obj/sm_<archs>/ and only rebuilt when
# their source is newer, since each FA2 translation unit takes minutes to compile.
# Writes the list of objects to link to obj/objects.txt.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
FA_ROOT="${FA_ROOT:-$HERE/../../third_party/flash-attention}"
source "$HERE/../cuda_arch.sh"

if [ -z "$FA2_ARCHS" ]; then
    OBJ_DIR="$HERE/obj/stub"
else
    OBJ_DIR="$HERE/obj/sm_${FA2_ARCHS// /_}"
fi
mkdir -p "$OBJ_DIR"

# Must match the head dims dispatched in fa2_api.cu.
HEAD_DIMS="32 64 96 128 192 256"
JOBS="${FA2_JOBS:-$(nproc)}"

if [ -z "$FA2_ARCHS" ]; then
    echo "No sm80+ target in CUDA_ARCHS=\"$CUDA_ARCHS\"; building without FlashAttention-2"
    NVCC_FLAGS=(-O3 -std=c++17 $(gencode_flags $CUDA_ARCHS) -DSDPA_FA2_DISABLED -Xcompiler -fPIC)
    SOURCES=("$HERE/fa2_api.cu")
else
    if [ ! -d "$FA_ROOT/csrc/cutlass/include" ]; then
        echo "FlashAttention sources not found at $FA_ROOT. Run:"
        echo "  git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git third_party/flash-attention"
        echo "  git -C third_party/flash-attention submodule update --init --depth 1 csrc/cutlass"
        exit 1
    fi

    NVCC_FLAGS=(
        -O3 -std=c++17
        $(gencode_flags $FA2_ARCHS)
        -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
        -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
        --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
        -DFLASH_NAMESPACE=sdpa_flash
        -DFLASHATTENTION_DISABLE_ALIBI -DFLASHATTENTION_DISABLE_SOFTCAP
        -Xcompiler -fPIC -Xcompiler -fvisibility=hidden
        -I"$HERE/torch_shim"
        -I"$FA_ROOT/csrc/flash_attn" -I"$FA_ROOT/csrc/flash_attn/src"
        -I"$FA_ROOT/csrc/cutlass/include"
    )

    SOURCES=("$HERE/fa2_api.cu")
    for hd in $HEAD_DIMS; do
        for dir in fwd bwd; do
            for dtype in fp16 bf16; do
                for causal in "" "_causal"; do
                    SOURCES+=("$FA_ROOT/csrc/flash_attn/src/flash_${dir}_hdim${hd}_${dtype}${causal}_sm80.cu")
                done
            done
        done
    done
fi

compile_one() {
    src="$1"
    obj="$OBJ_DIR/$(basename "${src%.cu}").o"
    # Up to date if newer than its source, this script and our headers
    # (fa2_api.h, torch_shim/). FA2's own headers are assumed unchanged.
    if [ "$obj" -nt "$src" ] && [ "$obj" -nt "$HERE/build_fa2.sh" ] \
        && [ -z "$(find "$HERE/torch_shim" "$HERE/fa2_api.h" -newer "$obj" -print -quit)" ]; then
        return 0
    fi
    echo "  nvcc $(basename "$src")"
    "$cudapath/bin/nvcc" -c "$src" -o "$obj" "${NVCC_FLAGS[@]}"
}
export OBJ_DIR HERE cudapath

if [ -n "$FA2_ARCHS" ]; then
    echo "Building FlashAttention-2 objects (archs: $FA2_ARCHS, head dims: $HEAD_DIMS)"
fi
# bash can't export arrays, so the flags and function are re-declared in each worker.
printf '%s\n' "${SOURCES[@]}" | xargs -P "$JOBS" -I{} \
    bash -ec "$(declare -p NVCC_FLAGS); $(declare -f compile_one); compile_one {}"

FA2_OBJS=()
for src in "${SOURCES[@]}"; do
    FA2_OBJS+=("$OBJ_DIR/$(basename "${src%.cu}").o")
done
printf '%s\n' "${FA2_OBJS[@]}" > "$HERE/obj/objects.txt"
