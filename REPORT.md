# GPU Programming Tutorial: Comprehensive Lab Report & Analysis
**Reference:** *Tutorial_GPU_Cuda.pdf*

---

## 1. Environment & Hardware Prerequisites

- **Detected GPU Model:** NVIDIA GeForce RTX 2060
- **Driver Version:** Windows WDDM 617.14 / WSL 615.78.02
- **CUDA Driver API Support:** CUDA 13.4
- **Global Memory:** 6.0 GiB GDDR6
- **Compute Capability:** 7.5 (Turing Architecture)
- **Compiler Status:** `nvcc` missing from system PATH (installation guidance provided).

---

## 2. Exercise 1 Analysis: Inspect the Available GPU

### Questions & Findings:
1. **Which reported limit constrains a 16-by-16-thread block?**
   - A $16 \times 16$ thread block has $16 \times 16 = 256$ threads.
   - On NVIDIA GPUs (including Compute Capability 7.5), the relevant hardware limits are:
     - `maxThreadsPerBlock`: 1024 threads.
     - `maxThreadsDim`: `[1024, 1024, 64]`.
   - Since $256 \le 1024$, and both dimension lengths ($16 \le 1024$) are within bounds, a 16-by-16 block easily satisfies the block limit. The primary constraint is that the total threads per block cannot exceed `maxThreadsPerBlock` (1024), and per-SM register and shared memory allocations must accommodate the active blocks.
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

### Speedup Definitions:
- **Resident Speedup:** $S_{\text{resident}} = \frac{T_{\text{CPU}}}{T_{\text{kernel}}}$
- **Operation Speedup:** $S_{\text{operation}} = \frac{T_{\text{CPU}}}{T_{\text{operation}}}$
- Where $T_{\text{operation}} = T_{\text{H2D}} + T_{\text{kernel}} + T_{\text{sync}} + T_{\text{D2H}}$.

### Behavior & Analysis:
- For small input sizes (e.g. $n = 1,003$), $S_{\text{operation}} < 1.0$ (negative speedup) because PCIe bus transfer latency (~5–15 $\mu$s) and CUDA runtime launch overhead dominate the minute compute time.
- As problem size grows to $n = 10,000,003$, GPU arithmetic throughput and memory parallelism saturate the GPU memory controllers, leading to substantial resident-data speedup ($S_{\text{resident}} \gg 1.0$).
- $S_{\text{resident}}$ measures compute efficiency assuming data already resides on device global memory (e.g. in multi-stage GPU pipelines), while $S_{\text{operation}}$ represents isolated end-to-end offloading costs.

---

## 6. Exercise 5 Analysis: Tune Launch Size and Memory Access

### Grid-Stride Loop:
$$\text{first} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}, \quad \text{stride} = \text{blockDim.x} \times \text{gridDim.x}$$
$$\text{Effective Bandwidth} = \frac{12 \times n}{T_{\text{kernel}} \times 10^9} \text{ GB/s}$$

### Insights:
1. **Coalesced Memory Access:** Adjacent threads within a warp read consecutive 32-bit floats (`a[i]`, `a[i+1]`), allowing the GPU memory controller to combine 32 loads into a single coalesced 128-byte DRAM transaction.
2. **Launch Configuration Sweeps:** Blocks (64, 128, 256) and Threads (64, 128, 256).
   - Configurations with total active threads matching or exceeding the GPU's SM capacity sustain peak memory bandwidth.
   - Overly small thread blocks (e.g. 64 threads = 2 warps) suffer from warp scheduler latency starvation, whereas 256 threads per block yields optimal occupancy.
3. **Single-thread grid-stride launch:**
   - A single thread with grid-stride loop can correctly compute all $n$ elements serially on the GPU, but it achieves negligible throughput because it underutilizes the thousands of parallel GPU cores.

---

## 7. Exercise 6 Analysis: Reduce a Vector with Shared Memory

### Halving Algorithm:
- Each block reduces 256 elements in `__shared__ int values[256]` via iterative halving (`offset = BLOCK / 2; offset > 0; offset /= 2`).
- Partial block sums are copied to `partial[blockIdx.x]`, and the final sum is aggregated.

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

### Naive vs Tiled Implementation:
- **Naive Kernel:** Every thread computes one element of $C$, performing $K$ global memory loads from $A$ and $K$ global memory loads from $B$. Total global memory accesses: $2 \times M \times N \times K$.
- **Tiled Kernel:** Threads collaboratively load $16 \times 16$ submatrices into fast `__shared__` memory tiles (`a_tile` and `b_tile`). Each element loaded from global memory is reused 16 times within the block's registers before the next tile is loaded, reducing global memory traffic by a factor of 16.
- **Two Barriers per Phase:**
  1. `__syncthreads()` #1: Ensures both tiles are fully loaded before threads compute partial dot products.
  2. `__syncthreads()` #2: Ensures all threads finish computing with the current tile before any thread overwrites the tile in the next phase.
- **Small Matrix Anomaly:** For very small matrices (e.g. $8 \times 8$), tiled multiplication can be slower than naive multiplication because shared memory allocation, loop overhead, and barrier synchronization latency outweigh the memory bandwidth savings.

---

## 9. Exercise 8 Analysis: Diagnostics with Compute Sanitizer

### 1. `vector_add_nobounds.cu` (Memcheck):
- Removing `if (i < n)` with $n=257$, `threads=256` results in 2 blocks ($2 \times 256 = 512$ threads).
- Threads $257 \dots 511$ access `d_a[i]`, `d_b[i]`, and `d_c[i]`.
- `compute-sanitizer --tool memcheck` immediately reports:
  `Invalid __global__ read of size 4 at <address> by thread (1,0,0) in block (1,0,0)`.

### 2. `reduce_sum_nobarrier.cu` (Racecheck):
- Removing the first `__syncthreads()` creates a Read-After-Write (RAW) / Write-After-Read (WAR) race hazard:
- Faster threads entering the reduction loop read `values[t + offset]` before slower threads have written `values[t] = a[i]`.
- `compute-sanitizer --tool racecheck` flags:
  `Hazard WAR/RAW on shared memory at values[t + offset] between threads`.
