#include "cuda_check.cuh"
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>

__global__ void vector_add(const float *a, const float *b,
                           float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char* argv[])
{
    int n = 1000003;
    int threads = 256;

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) threads = std::atoi(argv[2]);

    const int blocks = (n + threads - 1) / threads;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 97) * 0.25f;
        b[i] = (i % 31) * 0.5f;
    }

    // Host serial reference timed with std::chrono::steady_clock (T_CPU)
    using Clock = std::chrono::steady_clock;
    std::vector<float> reference(n);
    auto cpu_start = Clock::now();
    for (int i = 0; i < n; ++i) {
        reference[i] = a[i] + b[i];
    }
    double cpu_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - cpu_start).count();

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Untimed GPU warm-up launch
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Host chrono timer measuring transfer-inclusive operation (T_operation)
    auto wall_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    
    // CUDA events timing only kernel execution (T_kernel)
    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    double operation_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - wall_start).count();

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // Numerical validation
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        double expected = reference[i];
        if (!close_enough(c[i], expected)) ++errors;
    }

    double resident_speedup = (kernel_ms > 0.0f) ? (cpu_ms / kernel_ms) : 0.0;
    double operation_speedup = (operation_ms > 0.0) ? (cpu_ms / operation_ms) : 0.0;

    std::printf("CPU=%.6f ms kernel=%.6f ms operation=%.6f ms resident_speedup=%.3f operation_speedup=%.3f\n",
        cpu_ms, kernel_ms, operation_ms, resident_speedup, operation_speedup);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors != 0;
}
