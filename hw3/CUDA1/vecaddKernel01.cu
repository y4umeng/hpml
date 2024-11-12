__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    C[index] = A[index] + B[index];
}

