#include "matmultKernel.h"

#define FOOTPRINT_SIZE BLOCK_SIZE

__global__ void MatMulKernel(const Matrix* __restrict__ A, const Matrix* __restrict__ B, Matrix* __restrict__ C) {
    // Register variables for matrix indices
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Calculate starting index in C for the current thread's block
    float* Csub = &C->elements[C->stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];
    
    // Each thread computes one element of Csub in its copy of CValue
    float Cvalue = 0.0f;

    // Shared memory for tiles of A and B
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over sub-matrices of A and B required to compute Csub
    for (int m = 0; m < (A->width / BLOCK_SIZE); ++m) {

        // Load data from global memory into shared memory, each thread loads one element
        shared_A[thread_row][thread_col] = A->elements[(block_row * BLOCK_SIZE + thread_row) * A->stride + m * BLOCK_SIZE + thread_col];
        shared_B[thread_row][thread_col] = B->elements[(m * BLOCK_SIZE + thread_row) * B->stride + block_col * BLOCK_SIZE + thread_col];
        
        // Synchronize to make sure the whole tile is loaded
        __syncthreads();

        // Multiply the two tiles together and accumulate the results
        #pragma unroll
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += shared_A[thread_row][e] * shared_B[e][thread_col];
        }

        // Synchronize to make sure computation is done before loading new data
        __syncthreads();
    }

    // Write the computed value to the output matrix C
    Csub[thread_row * C->stride + thread_col] = Cvalue;
}
