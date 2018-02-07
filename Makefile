all: 	
	nvcc cuda/dwhjoin.cu -o bin/dwhjoin_cuda
	nvcc cuda_hello/hello.cu -o bin/cuda_hello
	gcc -Wall -fopenmp openmp/dwhjoin.c -o bin/dwhjoin_openmp
	gcc -Wall simple/dwhjoin.c -o bin/dwhjoin_simple