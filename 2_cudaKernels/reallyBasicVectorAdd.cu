#include <stdio.h>

#define N 1e7 //10mil

__global__ void cudaAdd(double* a, double* b, double* c) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id<N) {
        c[id] = a[id] + b[id];
    }
}

int main() {
    size_t bytes = N * sizeof(double);

    double* h_a = (double*)malloc(bytes);
    double* h_b = (double*)malloc(bytes);
    double* h_c = (double*)malloc(bytes);

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    for(int i=0; i<N; i++) {
        h_a[i] = (double)rand() / RAND_MAX;
        h_b[i] = (double)rand() / RAND_MAX;
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    cudaAdd<<<ceil(float(N)/256), 256>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("success!\n");

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}