#include "cuda_check.cuh"
#include <cstdio>
#include <cstdlib>

__global__ void hello(int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        printf("block=%u local=%u global=%d\n",
            blockIdx.x, threadIdx.x, i);
    }
}

int main(int argc, char* argv[])
{
    int n = 20;
    int threads = 8;
    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) threads = std::atoi(argv[2]);

    const int blocks = (n + threads - 1) / threads;
    hello<<<blocks, threads>>>(n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Host: the kernel has finished.\n");
    return 0;
}
