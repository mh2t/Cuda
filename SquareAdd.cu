#include <stdio.h>
#include <cuda.h>

#define M 4
#define N 6
#define MN (M*N)
#define BLOCK_SIZE 2
#define IDX(i,j) (i*N+j)

void initialize1(float *mat, int m, int n) {
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            mat[IDX(i,j)] = i+j;
        }
    }
}

void initialize2(float *mat, int m, int n) {
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            mat[IDX(i, j)] = i*j;
        }
    }
}

void printMatrix(float *mat, int m, int n) {
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            printf("%.4f ", mat[IDX(i,j)]);
        }
        printf("\n");
    }
} 
    
__global__ 

void matAdd(float *a, float *b, float *c) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
    int index = IDX(row, col);
    c[index] = a[index] + b[index];
    }
}

int main(void) {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = MN * sizeof(float);
    
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    h_a = (float *)malloc(size); initialize1(h_a, M, N);
    h_b = (float *)malloc(size); initialize2(h_b, M, N);
    h_c = (float *)malloc(size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    int m_blocks = (M+BLOCK_SIZE - 1) / BLOCK_SIZE;
    int n_blocks = (N+BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocksPerGrid(n_blocks, m_blocks);
    
    matAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    printf("A: \n"); printMatrix(h_a, M, N);
    printf("B: \n"); printMatrix(h_b, M, N);
    printf("C=A+B: \n"); printMatrix(h_c, M, N);
    
    return 0;
    
}