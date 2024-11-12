__global__ void AddVectors(const float* A, const float* B, float* C, int chunkSize)
{
    // Calculate starting index for the current thread's chunk
    int threadStartIndex = blockIdx.x * blockDim.x * chunkSize + threadIdx.x * chunkSize;

    // Process each element in the chunk assigned to this thread
    for (int i = 0; i < chunkSize; ++i) {
        int index = threadStartIndex + i;
        C[index] = A[index] + B[index];
    }
}

