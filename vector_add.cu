#include "cuda_check.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>

__global__ void vector_add(const float *a, const float *b,
                           float *c, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = alpha * a[i] + b[i];
}

int main(int argc, char* argv[])
{
    int n = 1000003;
    float alpha = 1.0f;
    int threads = 256;

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) alpha = static_cast<float>(std::atof(argv[2]));
    if (argc > 3) threads = std::atoi(argv[3]);

    const int blocks = (n + threads - 1) / threads;
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
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, alpha, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < n; ++i) {
        double expected = static_cast<double>(alpha) * static_cast<double>(a[i]) + static_cast<double>(b[i]);
        if (!close_enough(c[i], expected)) ++errors;
    }

    std::printf("n=%d blocks=%d threads=%d errors=%d\n",
        n, blocks, threads, errors);
    std::printf("first=%.2f last=%.2f\n", c.front(), c.back());

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors != 0;
}
