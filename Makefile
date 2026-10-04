NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo -arch=native

TARGETS = device_info \
          hello_gpu \
          vector_add \
          vector_benchmark \
          vector_stride \
          reduce_sum \
          matrix_mul \
          matrix_mul_tiled \
          vector_add_nobounds \
          reduce_sum_nobarrier

.PHONY: all clean

all: $(TARGETS)

%: %.cu cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
