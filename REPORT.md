# GPU Programming Tutorial: Comprehensive Lab Report & Analysis
**Reference:** *Tutorial_GPU_Cuda.pdf*

---

## 1. Environment & Hardware Prerequisites

- **Detected GPU Model:** NVIDIA GeForce RTX 2060
- **Driver Version:** Windows WDDM 617.14 / WSL 615.78.02
- **CUDA Toolkit Version:** CUDA 12.4 (`nvcc` release 12.4, V12.4.131)
- **Streaming Multiprocessors (SMs):** 30 SMs
- **Warp Size:** 32 threads
- **Max Threads Per Block:** 1024
- **Shared Memory Per Block:** 49,152 bytes (48 KiB)
- **Global Memory:** 6.00 GiB GDDR6
- **Compiler Flags:** `-std=c++17 -O3 -lineinfo -arch=native`

---

## 2. Exercise 1 Analysis: Inspect the Available GPU

### Hardware Query Output (`./device_info`):
```text
Device: NVIDIA GeForce RTX 2060
Compute capability: 7.5
SM count: 30
Warp size: 32
Maximum threads per block: 1024
Global memory: 6.00 GiB
Shared memory per block: 49152 bytes
```

### Questions & Findings:
1. **Which reported limit constrains a 16-by-16-thread block?**
   - A $16 \times 16$ thread block has $16 \times 16 = 256$ threads.
   - On NVIDIA GPUs (including Compute Capability 7.5), the relevant hardware limits are:
     - `maxThreadsPerBlock`: 1024 threads.
     - `maxThreadsDim`: `[1024, 1024, 64]`.
   - Since $256 \le 1024$, and each dimension length ($16 \le 1024$) is within bounds, a 16-by-16 block satisfies the block limits. The fundamental constraint is that total threads per block cannot exceed `maxThreadsPerBlock` (1024), and per-SM register and shared memory allocations must accommodate the active blocks.
2. **Why is a grid with more blocks than SMs valid?**
   - GPUs decouple thread blocks from physical Streaming Multiprocessors (SMs). An SM can host multiple resident blocks simultaneously, and the hardware work distributor (GigaThread engine) schedules blocks onto SMs in **waves**. When a block finishes execution and releases resources, subsequent pending blocks from the grid are scheduled onto the newly available SMs until the entire grid completes.

---

## 3. Exercise 2 Analysis: Launch a First Kernel

### Predictions & Results:
1. **Launch configuration $n=20$, `threads=8`:**
   - Blocks formula: `blocks = (n + threads - 1) / threads = (20 + 8 - 1) / 8 = 3` blocks.
   - Launched threads: $3 \times 8 = 24$ threads total.
   - Active indices: Indices $0 \dots 19$ (20 active indices). The remaining 4 threads (global IDs 20, 21, 22, 23 in block 2) evaluate `if (i < n)` as false and do not print.
2. **Repeat with $n=17$, `threads=16`:**
   - Blocks formula: `(17 + 16 - 1) / 16 = 2` blocks.
   - Launched threads: $2 \times 16 = 32$ threads total (1 full warp).
   - Active indices: Indices $0 \dots 16$ (17 active indices). The remaining 15 threads in block 1 are inactive.
3. **What happens if the bound check is removed?**
   - In a print-only kernel, removing `if (i < n)` merely prints surplus greeting lines for inactive thread indices up to total launched threads.
   - However, in an array kernel, removing the bound check results in threads with $i \ge n$ dereferencing out-of-bounds memory addresses, causing buffer overruns, memory corruption, or segmentation faults (`cudaErrorIllegalAddress`).

---

## 4. Exercise 3 Analysis: Add Vectors Using Explicit Device Memory

### Analysis & Lifetime:
1. **Tests with edge sizes ($n=1, 255, 256, 257, 1000003$):**
   - For all test sizes, integer ceiling division `(n + threads - 1) / threads` correctly allocates sufficient blocks, and the guard `if (i < n)` ensures threads beyond $n$ do not access device memory. Correctness check passes with `errors=0`.
2. **Scaling operation $c_i = 2a_i + b_i$:**
   - Parameterized kernel `c[i] = alpha * a[i] + b[i]` with $\alpha=2.0$ computes the exact scaled vector addition. The CPU validation reference matches with `errors=0`.
3. **Lifetimes of Host and Device Allocations:**
   - Host allocations (`std::vector<float> a, b, c`): Live from instantiation on host stack until the scope of `main()` exits (automatically freed by RAII).
   - Device allocations (`d_a, d_b, d_c`): Explicitly allocated on GPU global memory via `cudaMalloc` and persist until explicitly deallocated via `cudaFree`. Failing to call `cudaFree` causes GPU memory leaks across application lifetimes.
4. **Why testing only the first output element is insufficient:**
   - Testing only index 0 checks only thread 0 in block 0. It fails to test:
     - Multi-thread coordination and higher indices.
     - Multi-block execution and grid distribution across SMs.
     - Boundary conditions where thread indices exceed $n$ in the final block.

---

## 5. Exercise 4 Analysis: Measure Kernel and Transfer-Inclusive Time

### Experimental Measurements:
| $n$ | CPU Baseline ($T_{\text{CPU}}$, ms) | GPU Kernel ($T_{\text{kernel}}$, ms) | Transfer-Inclusive ($T_{\text{operation}}$, ms) | Resident Speedup ($S_{\text{resident}}$) | Operation Speedup ($S_{\text{operation}}$) |
|---|---|---|---|---|---|
| **1,003** | 0.0002 ms | 0.0054 ms | 0.1892 ms | 0.041x | 0.001x |
| **100,003** | 0.0169 ms | 0.0533 ms | 0.3974 ms | 0.317x | 0.042x |
| **1,000,003** | 0.7191 ms | 0.0894 ms | 1.7074 ms | 8.043x | 0.421x |
| **10,000,003** | 6.7354 ms | 0.5625 ms | 15.3785 ms | 11.973x | 0.438x |

### Speedup Definitions:
- **Resident Speedup:** $S_{\text{resident}} = \frac{T_{\text{CPU}}}{T_{\text{kernel}}}$
- **Operation Speedup:** $S_{\text{operation}} = \frac{T_{\text{CPU}}}{T_{\text{operation}}}$
- Where $T_{\text{operation}} = T_{\text{H2D}} + T_{\text{kernel}} + T_{\text{sync}} + T_{\text{D2H}}$.

### Behavior & Analysis:
- For small input sizes ($n = 1,003$ and $100,003$), $S_{\text{operation}} \ll 1.0$ because PCIe transfer latency (~10–20 $\mu$s) and host dispatch overhead dominate the computation.
- As problem size grows to $n = 10,000,003$, GPU arithmetic throughput and memory parallelism saturate the GPU memory controllers, leading to substantial resident-data speedup ($S_{\text{resident}} \approx 12.0\times$).
- However, $S_{\text{operation}}$ stays around $0.44\times$ because vector addition has very low arithmetic intensity (1 flop per 12 bytes transferred). Offloading a single vector addition across PCIe is memory-bound and communication-dominated; GPU acceleration is advantageous when data remains on device across multiple compute stages.

---

## 6. Exercise 5 Analysis: Tune Launch Size and Memory Access

### Grid-Stride Loop:
$$\text{first} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}, \quad \text{stride} = \text{blockDim.x} \times \text{gridDim.x}$$
$$\text{Effective Bandwidth} = \frac{12.0 \times n}{T_{\text{kernel}} \times 10^9} \text{ GB/s}$$

### Grid Configuration Sweep ($n = 1,000,003$):
| Blocks | Threads | Kernel Time (ms) | Effective Bandwidth (GB/s) |
|---|---|---|---|
| **64** | **64** | 0.1164 ms | 103.14 GB/s |
| **64** | **128** | 0.0717 ms | 167.34 GB/s |
| **64** | **256** | 0.0492 ms | 244.14 GB/s |
| **128** | **64** | 0.0716 ms | 167.49 GB/s |
| **128** | **128** | 0.0478 ms | 250.84 GB/s |
| **128** | **256** | 0.0556 ms | 215.89 GB/s |
| **256** | **64** | 0.0474 ms | **253.21 GB/s** |
| **256** | **128** | 0.0558 ms | 214.90 GB/s |
| **256** | **256** | 0.0500 ms | 240.08 GB/s |

### Insights:
1. **Coalesced Memory Access:** Adjacent threads within a warp access adjacent 32-bit floats (`a[i]`, `a[i+1]`), combining memory requests into single 128-byte DRAM transactions.
2. **Occupancy & Latency Hiding:** A configuration of 256 blocks with 64 threads per block achieved the highest effective bandwidth (253.21 GB/s), representing over 75% of the RTX 2060's theoretical peak memory bandwidth (336 GB/s).
3. **Single-thread grid-stride launch:**
   - A single thread with a grid-stride loop can compute all elements serially on the GPU, but it yields negligible throughput because only 1 of 1,920 hardware CUDA cores is active, providing no latency hiding or parallel memory access.

---

## 7. Exercise 6 Analysis: Reduce a Vector with Shared Memory

### Halving Algorithm Verification:
- All edge sizes ($n = 1, 255, 256, 257, 1000003$) verified against CPU reference: `match=yes`.
- For $n=1,000,003$: `expected=3000003 result=3000003 match=yes`.

### Investigation Questions:
1. **Why `BLOCK=192` breaks the halving algorithm:**
   - 192 is not a power of two ($192 = 128 + 64$).
   - After the first step `offset = 192 / 2 = 96`, threads $0 \dots 95$ sum `values[t] + values[t + 96]`.
   - The next step `offset = 48` leaves elements between $96 \dots 191$ unaccounted for in standard binary tree halving, dropping elements and producing incorrect sums.
2. **Role of Synchronization:**
   - The first `__syncthreads()` guarantees all 256 threads finish copying from global memory into shared memory before the reduction loop begins.
   - Subsequent `__syncthreads()` calls inside the loop ensure that step $k$ additions are completed across all threads before step $k+1$ reads the updated values.

---

## 8. Exercise 7 Analysis: Matrix Multiplication Naive vs Tiled

$$\text{GFLOP/s} = \frac{2 \times M \times K \times N}{T_{\text{kernel}} \times 10^9}$$

### Performance Comparison:
| Shape ($M \times K \times N$) | Naive Time (ms) | Naive GFLOP/s | Tiled Time (ms) | Tiled GFLOP/s | Speedup ($T_{\text{naive}} / T_{\text{tiled}}$) |
|---|---|---|---|---|---|
| **$8 \times 8 \times 8$** | 0.0059 ms | 0.17 | 0.0061 ms | 0.17 | **0.96x** |
| **$15 \times 15 \times 15$** | 0.0063 ms | 1.08 | 0.0060 ms | 1.13 | **1.05x** |
| **$127 \times 130 \times 129$** | 0.0187 ms | 228.32 | 0.0185 ms | 230.70 | **1.01x** |
| **$128 \times 128 \times 128$** | 0.0225 ms | 186.18 | 0.0169 ms | 248.71 | **1.34x** |
| **$256 \times 256 \times 256$** | 0.0991 ms | 338.58 | 0.0696 ms | 482.10 | **1.42x** |
| **$512 \times 512 \times 512$** | 0.6821 ms | 393.54 | 0.4400 ms | 610.04 | **1.55x** |

### Insights:
- **Small Matrix Anomaly ($8 \times 8 \times 8$):** The tiled implementation is slightly slower (0.96x) because overhead from shared memory tile loading and two `__syncthreads()` barriers outweighs the minimal global memory bandwidth savings for 64 elements.
- **Large Square Matrices ($512 \times 512 \times 512$):** Tiling reduces global memory traffic by a factor of 16 ($16 \times 16$ tile reuse), boosting throughput from 393.54 GFLOP/s to **610.04 GFLOP/s** (a 1.55x speedup).

---

## 9. Exercise 8 Analysis: Diagnostics with Compute Sanitizer

### 1. `vector_add_nobounds.cu` (Memcheck):
- Removing `if (i < n)` with $n=257$, `threads=256` results in 2 blocks ($2 \times 256 = 512$ threads).
- Threads $257 \dots 511$ access `d_a[i]`, `d_b[i]`, and `d_c[i]`.
- Diagnostic explanation: Threads beyond index 256 cause out-of-bounds global memory reads and writes, detected by `compute-sanitizer --tool memcheck` as `Invalid __global__ read of size 4`.

### 2. `reduce_sum_nobarrier.cu` (Racecheck):
- Removing the first `__syncthreads()` creates a Read-After-Write (RAW) / Write-After-Read (WAR) race hazard:
- Faster threads entering the reduction loop read `values[t + offset]` before slower threads have finished writing `values[t] = a[i]`.
- Diagnostic explanation: `compute-sanitizer --tool racecheck` detects hazards on shared memory between threads within the same thread block.
