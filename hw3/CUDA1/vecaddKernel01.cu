#include <stdio.h>

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int threadStartIndex = blockStartIndex + (threadIdx.x);
    int threadEndIndex   = threadStartIndex + blockDim.x * N;
    int i;

    for( i=threadStartIndex; i<threadEndIndex; i+=blockDim.x){
        C[i] = A[i] + B[i];
    }
}
