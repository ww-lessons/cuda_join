all: 
	cp -f cuda/dwhjoin.c cuda/dwhjoin.cu
	nvcc cuda/dwhjoin.cu -o bin/dwhjoin_cuda
	gcc -Wall -fopenmp openmp/dwhjoin.c -o bin/dwhjoin_openmp
	gcc -Wall simple/dwhjoin.c -o bin/dwhjoin_simple