#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000 //10mil element array
#define BLOCK_SIZE 256

//normal (cringe) CPU vector addition
void addCPU(float* a, float* b, float* c, int n) {
    for(int i=0; i<n; i++) {
        c[i] = a[i] + b[i];
    }
}

//swag CUDA kernel to parallelise it
__global__ void addGPU(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<n) {
        c[i] = a[i] + b[i];
    }
}

//init vector with random values
void init_vector(float* vec, int n) {
    for(int i=0; i<n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

//function to get execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts); //timespec structure returns counter value in nanoseconds
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_cCPU, *h_cGPU;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    //dynamic allocation of memory to arrays in C
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_cCPU = (float*)malloc(size);
    h_cGPU = (float*)malloc(size);

    //init vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    //allocate memory to gpu
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    //copy arrays to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    //define dimensions
    int num_blocks = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    //this is equal to roughly 4 if N=1024

    //cpu run
    printf("Running CPU function....\n");
    double cpuTime = 0.0;
    for(int i=0; i<20; i++) {
        double startTime = get_time();
        addCPU(h_a, h_b, h_cCPU, N);
        double endTime = get_time();
        cpuTime += endTime-startTime;
    }
    double cpuAvgTime = cpuTime/20.0;

    //gpu run
    printf("Running GPU kernel....\n");
    double gpuTime = 0.0;
    for(int i=0; i<20; i++) {
        double startTime = get_time();
        addGPU<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double endTime = get_time();
        gpuTime += endTime-startTime;
    }
    double gpuAvgTime = gpuTime/20.0;


    printf("Cpu time: %f ms.\n", cpuAvgTime*1000);
    printf("Gpu time: %f ms.\n", gpuAvgTime*1000);
    printf("Speedup: %fx\n", cpuAvgTime/gpuAvgTime);

    //ffreeeeee memoryyyyy
    free(h_a);
    free(h_b);
    free(h_cCPU);
    free(h_cGPU);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;
}

