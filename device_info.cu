#include "cuda_check.cuh"
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[])
{
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        std::fprintf(stderr, "No CUDA device is available.\n");
        return 1;
    }

    int dev = 0;
    if (argc > 1) {
        dev = std::atoi(argv[1]);
    }

    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    std::printf("Device: %s\n", p.name);
    std::printf("Compute capability: %d.%d\n", p.major, p.minor);
    std::printf("SM count: %d\n", p.multiProcessorCount);
    std::printf("Warp size: %d\n", p.warpSize);
    std::printf("Maximum threads per block: %d\n", p.maxThreadsPerBlock);
    std::printf("Global memory: %.2f GiB\n",
        p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("Shared memory per block: %zu bytes\n",
        static_cast<std::size_t>(p.sharedMemPerBlock));
    return 0;
}
