__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int blockEndIndex = blackStartIndex + 32 * N;
    // int threadStartIndex = blockStartIndex + (threadIdx.x * N);
    // int threadEndIndex   = threadStartIndex + N;
    int i;
    int stride = N;

    for( i=blockStartIndex; i<blockEndIndex; i += stride){
        C[i] = A[i] + B[i];
    }
}

