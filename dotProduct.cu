#include <stdio.h>
#include <cuda.h>

#define N 6

void initialize(float * vector, int num) {
    for(int i=0; i<num; i++) {
        vector[i] = i;
    }
}    

void plotVector(float * vector, int num) {
    for(int i=0; i<num; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}

__global__ void dot_product(float *a, float *b, float *c) 
    {
    c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
    }
    
int main(void) 
    {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = N * sizeof(float);
    
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    h_a = (float *)malloc(size); initialize(h_a, N);
    h_b = (float *)malloc(size); initialize(h_b, N);
    h_c = (float *)malloc(size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    dot_product<<<1,N>>>(d_a, d_b, d_c);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    float sum = 0;
    for(int i=0; i<N; i++) {
        sum += h_c[i];
    }
    
    printf("Vector A: \n"); plotVector(h_a, N);
    printf("Vector B: \n"); plotVector(h_b, N);
    printf("Dot Product, A*B = %f\n", sum); 
    
    return 0;
    }