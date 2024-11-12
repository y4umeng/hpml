__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    // Calculate the global index of the current thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Use a stride to allow threads to handle the entire vector
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements spaced apart by the grid stride
    for (int i = index; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
