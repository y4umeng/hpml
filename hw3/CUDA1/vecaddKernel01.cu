#include <stdio.h>

// __global__ void AddVectors(const float* A, const float* B, float* C, int N)
// {
//     int blockStartIndex  = blockIdx.x * blockDim.x * N;
//     int threadStartIndex = blockStartIndex + threadIdx.x;
//     int threadEndIndex   = threadStartIndex + blockDim.x * N;
//     int i;

//     for( i=threadStartIndex; i<=threadEndIndex; i+=N ){
//         C[i] = A[i] + B[i];
//     }
// }

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each value assigned to this thread
    for (int i = 0; i < N; i += blockDim.x * gridDim.x) {
        int idx = index + i;
        if (idx < N) {  // Boundary check
            C[idx] = A[idx] + B[idx];
        }
    }
}
