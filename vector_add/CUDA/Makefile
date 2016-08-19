NVCC = nvcc
NVFLAGS = -std=c++11 -g -O2

.default: parallel_vector_add

parallel_vector_add:
	$(NVCC) $(NVFLAGS) -o parallel_vector_add parallel_vector_add.cu


clean:
	rm parallel_vector_add
