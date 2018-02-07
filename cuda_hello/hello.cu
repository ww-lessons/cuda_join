#include <stdio.h>

__global__ void get_hello(int *i) {
    i[0] = 123;
}

int main(int argc, char *argv[]) {
    int *buf_device;
    int *buf_host;

    buf_host = (int*)calloc(sizeof(int), 1);
    cudaMalloc(&buf_device, sizeof(int));

    cudaMemcpy(buf_device, buf_host, sizeof(int), cudaMemcpyHostToDevice);
    get_hello<<<1, 1>>>(buf_device);

    cudaMemcpy(buf_host, buf_device, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Ergebnis: %d\n", buf_host[0]);

    cudaFree(&buf_device);
    free(buf_host);    
    return 0;
}