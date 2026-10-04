#!/usr/bin/env bash
set -euo pipefail

NVCC=${NVCC:-nvcc}
NVCCFLAGS="-std=c++17 -O3 -lineinfo -arch=native"

TARGETS=(
    "device_info"
    "hello_gpu"
    "vector_add"
    "vector_benchmark"
    "vector_stride"
    "reduce_sum"
    "matrix_mul"
    "matrix_mul_tiled"
    "vector_add_nobounds"
    "reduce_sum_nobarrier"
)

echo "============================================="
echo "Building CUDA GPU Lab Assignment Programs"
echo "Compiler: $NVCC"
echo "Flags: $NVCCFLAGS"
echo "============================================="

for target in "${TARGETS[@]}"; do
    echo "Compiling ${target}.cu -> ${target}..."
    $NVCC $NVCCFLAGS "${target}.cu" -o "$target"
done

echo "============================================="
echo "All targets compiled successfully!"
echo "============================================="
