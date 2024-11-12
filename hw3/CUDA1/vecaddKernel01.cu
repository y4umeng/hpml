__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int index = blockIdx.x * blockDim.x * N + threadIdx.x;

    // Loop with stride of N
    for (int i = index; i < N * N; i += N) {
        C[i] = A[i] + B[i];
    }
}

