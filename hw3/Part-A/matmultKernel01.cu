#include "matmultKernel.h"

#define FOOTPRINT_SIZE 32

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Define row and column indices for each thread
    int thread_row = threadIdx.y * 2;  // Each thread computes two rows
    int thread_col = threadIdx.x * 2;  // Each thread computes two columns
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Calculate starting index in C for the current thread's block
    float* Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];
    
    // Each thread computes four elements of Csub in its copy of Cvalue
    float Cvalue[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    // Shared memory for tiles of A and B
    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Loop over tiles of A and B
    for (int m = 0; m < (A.width / FOOTPRINT_SIZE); ++m) {
        // Load 4 elements of A and B into shared memory in coalesced manner
        int A_row = block_row * FOOTPRINT_SIZE + thread_row;
        int A_col = m * FOOTPRINT_SIZE + thread_col;
        int B_row = m * FOOTPRINT_SIZE + thread_row;
        int B_col = block_col * FOOTPRINT_SIZE + thread_col;

        shared_A[thread_row][thread_col]     = A.elements[A_row * A.stride + A_col];
        shared_A[thread_row][thread_col + 1] = A.elements[A_row * A.stride + A_col + 1];
        shared_A[thread_row + 1][thread_col] = A.elements[(A_row + 1) * A.stride + A_col];
        shared_A[thread_row + 1][thread_col + 1] = A.elements[(A_row + 1) * A.stride + A_col + 1];

        shared_B[thread_row][thread_col]     = B.elements[B_row * B.stride + B_col];
        shared_B[thread_row][thread_col + 1] = B.elements[B_row * B.stride + B_col + 1];
        shared_B[thread_row + 1][thread_col] = B.elements[(B_row + 1) * B.stride + B_col];
        shared_B[thread_row + 1][thread_col + 1] = B.elements[(B_row + 1) * B.stride + B_col + 1];

        // Synchronize to make sure the whole tile is loaded
        __syncthreads();

        // Unroll the inner loop to compute the 2x2 output sub-block
        #pragma unroll
        for (int e = 0; e < FOOTPRINT_SIZE; ++e) {
            Cvalue[0][0] += shared_A[thread_row][e] * shared_B[e][thread_col];
            Cvalue[0][1] += shared_A[thread_row][e] * shared_B[e][thread_col + 1];
            Cvalue[1][0] += shared_A[thread_row + 1][e] * shared_B[e][thread_col];
            Cvalue[1][1] += shared_A[thread_row + 1][e] * shared_B[e][thread_col + 1];
        }

        // Synchronize to ensure all threads have finished using shared memory
        __syncthreads();
    }

    // Write the computed values to global memory in a coalesced manner
    Csub[thread_row * C.stride + thread_col] = Cvalue[0][0];
    Csub[thread_row * C.stride + thread_col + 1] = Cvalue[0][1];
    Csub[(thread_row + 1) * C.stride + thread_col] = Cvalue[1][0];
    Csub[(thread_row + 1) * C.stride + thread_col + 1] = Cvalue[1][1];
}
