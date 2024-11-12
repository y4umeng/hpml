__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't access out-of-bounds elements
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

