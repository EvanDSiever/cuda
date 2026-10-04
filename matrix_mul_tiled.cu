#include "cuda_check.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

constexpr int TILE = 16;

__global__ void matmul(const float *a, const float *b, float *c,
                       int m, int k, int n)
{
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;
    float sum = 0.0f;

    for (int base = 0; base < k; base += TILE) {
        int a_col = base + tx;
        int b_row = base + ty;
        a_tile[ty][tx] = (row < m && a_col < k)
            ? a[row * k + a_col] : 0.0f;
        b_tile[ty][tx] = (b_row < k && col < n)
            ? b[b_row * n + col] : 0.0f;
        __syncthreads();
        for (int q = 0; q < TILE; ++q) {
            sum += a_tile[ty][q] * b_tile[q][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = sum;
}

int main(int argc, char* argv[])
{
    int m = 127, k = 130, n = 129;
    if (argc > 1) m = std::atoi(argv[1]);
    if (argc > 2) k = std::atoi(argv[2]);
    if (argc > 3) n = std::atoi(argv[3]);

    std::vector<float> a(m * k), b(k * n), c(m * n);
    for (int i = 0; i < m * k; ++i) a[i] = ((i % 17) - 8) * 0.125f;
    for (int i = 0; i < k * n; ++i) b[i] = ((i % 13) - 6) * 0.0625f;

    std::size_t a_bytes = a.size() * sizeof(float);
    std::size_t b_bytes = b.size() * sizeof(float);
    std::size_t c_bytes = c.size() * sizeof(float);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), a_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), b_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), c_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((n + block.x - 1) / block.x,
              (m + block.y - 1) / block.y);

    // Warm-up launch outside timed interval
    matmul<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Timed kernel execution
    CUDA_CHECK(cudaEventRecord(start));
    matmul<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    double max_error = 0.0;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double reference = 0.0;
            for (int q = 0; q < k; ++q) {
                reference += static_cast<double>(a[row * k + q]) *
                             b[q * n + col];
            }
            double got = c[row * n + col];
            double error = std::fabs(got - reference);
            if (error > max_error) max_error = error;
            if (!close_enough(got, reference, 1e-4, 1e-4)) ++errors;
        }
    }

    double kernel_seconds = kernel_ms / 1000.0;
    double gflops = (2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n)) / (kernel_seconds * 1e9);

    std::printf("M=%d K=%d N=%d errors=%d max_error=%.3e kernel_time=%.6f ms GFLOP/s=%.3f\n",
        m, k, n, errors, max_error, kernel_ms, gflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors != 0;
}
