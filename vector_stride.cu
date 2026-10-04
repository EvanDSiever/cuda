#include "cuda_check.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>

__global__ void vector_add_stride(const float *a, const float *b,
                                  float *c, int n)
{
    int first = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = first; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[])
{
    int n = 1000003;
    int blocks = 128;
    int threads = 256;

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) blocks = std::atoi(argv[2]);
    if (argc > 3) threads = std::atoi(argv[3]);

    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 97) * 0.25f;
        b[i] = (i % 31) * 0.5f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // Warm-up launch
    vector_add_stride<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Timed kernel execution
    CUDA_CHECK(cudaEventRecord(start));
    vector_add_stride<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < n; ++i) {
        double expected = static_cast<double>(a[i]) + static_cast<double>(b[i]);
        if (!close_enough(c[i], expected)) ++errors;
    }

    double kernel_seconds = kernel_ms / 1000.0;
    double bandwidth_gb_s = (12.0 * static_cast<double>(n)) / (kernel_seconds * 1e9);

    std::printf("n=%d blocks=%d threads=%d errors=%d kernel_time=%.6f ms Bandwidth=%.3f GB/s\n",
        n, blocks, threads, errors, kernel_ms, bandwidth_gb_s);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors != 0;
}
