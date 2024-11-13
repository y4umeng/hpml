#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define H 1024
#define W 1024
#define C 3
#define K 64
#define FH 3
#define FW 3
#define P 1

__global__ void tiledConvolutionKernel(const double* I0, const double* F, double* O, int width, int height) {
    extern __shared__ double tile[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    if (x < width && y < height) {
        double result = 0.0;
        
        for (int c = 0; c < C; ++c) {
            // Load input tile with padding into shared memory
            int input_x = x - P;
            int input_y = y - P;
            if (input_x >= 0 && input_x < width + 2 * P && input_y >= 0 && input_y < height + 2 * P) {
                tile[threadY * blockDim.x + threadX] = I0[c * (width + 2 * P) * (height + 2 * P) + input_y * (width + 2 * P) + input_x];
            } else {
                tile[threadY * blockDim.x + threadX] = 0.0;
            }

            __syncthreads();

            // Perform convolution
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    int tx = FW - 1 - i;
                    int ty = FH - 1 - j;
                    int ix = threadX + i;
                    int iy = threadY + j;

                    if (ix < blockDim.x && iy < blockDim.y) {
                        result += F[(k * C + c) * FH * FW + tx * FW + ty] * tile[iy * blockDim.x + ix];
                    }
                }
            }

            __syncthreads();
        }
        
        O[k * width * height + y * width + x] = result;
    }
}

void initializeInput(double* I) {
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
                I[c * H * W + x * W + y] = c * (x + y);
            }
        }
    }
}

void initializeFilter(double* F) {
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }
}

void addPadding(const double* I, double* I0) {
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H + 2 * P; ++x) {
            for (int y = 0; y < W + 2 * P; ++y) {
                if (x < P || x >= H + P || y < P || y >= W + P) {
                    I0[c * (H + 2 * P) * (W + 2 * P) + x * (W + 2 * P) + y] = 0.0;
                } else {
                    I0[c * (H + 2 * P) * (W + 2 * P) + x * (W + 2 * P) + y] = I[c * H * W + (x - P) * W + (y - P)];
                }
            }
        }
    }
}

int main() {
    size_t imageSize = C * H * W * sizeof(double);
    size_t paddedSize = C * (H + 2 * P) * (W + 2 * P) * sizeof(double);
    size_t filterSize = K * C * FH * FW * sizeof(double);
    size_t outputSize = K * H * W * sizeof(double);

    double *I, *I0, *F, *O;
    cudaMallocManaged(&I, imageSize);
    cudaMallocManaged(&I0, paddedSize);
    cudaMallocManaged(&F, filterSize);
    cudaMallocManaged(&O, outputSize);

    initializeInput(I);
    initializeFilter(F);
    addPadding(I, I0);

    dim3 blockSize(16, 16);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y, K);
    size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(double);

    auto start = std::chrono::high_resolution_clock::now();
    tiledConvolutionKernel<<<gridSize, blockSize, sharedMemSize>>>(I0, F, O, W, H);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Kernel execution time: " << elapsed.count() << " seconds" << std::endl;

    double checksum = 0.0;
    for (int i = 0; i < K * W * H; ++i) {
        checksum += O[i];
    }
    std::cout << "Checksum (sum of all elements in O): " << checksum << std::endl;

    cudaFree(I);
    cudaFree(I0);
    cudaFree(F);
    cudaFree(O);

    return 0;
}
