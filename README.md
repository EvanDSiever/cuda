# CUDA GPU Lab Assignment: Complete Implementation & Guide
Based on *Tutorial_GPU_Cuda.pdf* (Exercises 1–8)

## 1. Overview & Directory Structure

All source files have been generated using modern C++17 and `cuda_check.cuh`. Every program is parameterized to accept command-line arguments for problem sizes, block counts, and thread counts while defaulting to the PDF's primary values.

### Files Created:
| File | Role / Exercise | Default CLI Invocation | Parameter Options |
|---|---|---|---|
| `cuda_check.cuh` | Shared Support Header | Included by all `.cu` files | `CUDA_CHECK(...)`, `close_enough(...)` |
| `device_info.cu` | Exercise 1: Device Inspection | `./device_info` | `[device_id=0]` |
| `hello_gpu.cu` | Exercise 2: Launch & Thread Mapping | `./hello_gpu 20 8` | `[n=20] [threads=8]` |
| `vector_add.cu` | Exercise 3: Explicit Memory & Vector Add | `./vector_add 1000003 1.0 256` | `[n=1000003] [alpha=1.0] [threads=256]` |
| `vector_benchmark.cu` | Exercise 4: Kernel & Operation Timing | `./vector_benchmark 1000003 256` | `[n=1000003] [threads=256]` |
| `vector_stride.cu` | Exercise 5: Grid-Stride Loop & Bandwidth | `./vector_stride 1000003 128 256` | `[n=1000003] [blocks=128] [threads=256]` |
| `reduce_sum.cu` | Exercise 6: Shared Memory Vector Reduction | `./reduce_sum 1000003` | `[n=1000003]` |
| `matrix_mul.cu` | Exercise 7: Naive 2D Matrix Multiplication | `./matrix_mul 127 130 129` | `[M=127] [K=130] [N=129]` |
| `matrix_mul_tiled.cu` | Exercise 7: Tiled Shared-Memory MatMul | `./matrix_mul_tiled 127 130 129` | `[M=127] [K=130] [N=129]` |
| `vector_add_nobounds.cu` | Exercise 8: Buggy Copy (No Bounds Guard) | `./vector_add_nobounds 257 256` | `[n=257] [threads=256]` |
| `reduce_sum_nobarrier.cu` | Exercise 8: Buggy Copy (Missing Barrier) | `./reduce_sum_nobarrier 1000003` | `[n=1000003]` |
| `Makefile` | Phase 3: Build Automation | `make all` | Standard `nvcc` compilation |
| `build.sh` | Phase 3: Shell Build Automation | `./build.sh` | Automated compilation script |
| `benchmark_and_plot.py` | Automated Benchmarking & Plotting | `python3 benchmark_and_plot.py` | Runs sweeps & plots figures |
| `REPORT.md` | Comprehensive Lab Report & Analysis | N/A | Full answers to Exercises 1-8 |

---

## 2. Compilation Instructions

Compile all programs with:
```bash
make all
# or
./build.sh
```

Individual compilation command syntax:
```bash
nvcc -std=c++17 -O3 -lineinfo -arch=native <file>.cu -o <file>
```

---

## 3. Running & Automated Benchmarking

Run individual exercises:
```bash
# Exercise 1
./device_info

# Exercise 2
./hello_gpu 20 8
./hello_gpu 17 16

# Exercise 3
./vector_add 1000003 1.0 256
./vector_add 1000003 2.0 256

# Exercise 4
./vector_benchmark 1000003 256

# Exercise 5
./vector_stride 1000003 128 256

# Exercise 6
./reduce_sum 1000003

# Exercise 7
./matrix_mul 127 130 129
./matrix_mul_tiled 127 130 129

# Exercise 8 Sanitizer Diagnostics
compute-sanitizer --tool memcheck ./vector_add_nobounds 257 256
compute-sanitizer --tool racecheck ./reduce_sum_nobarrier 1000003
```

To automatically benchmark, verify, and generate speedup and bandwidth plots:
```bash
python3 benchmark_and_plot.py
```
