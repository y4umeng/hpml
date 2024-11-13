#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <chrono>

#define H 1024
#define W 1024
#define C 3
#define K 64
#define FH 3
#define FW 3
#define P 1

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

int main() {
    size_t imageSize = C * H * W * sizeof(double);
    size_t filterSize = K * C * FH * FW * sizeof(double);
    size_t outputSize = K * H * W * sizeof(double);

    double *I, *F, *O;
    cudaMallocManaged(&I, imageSize);
    cudaMallocManaged(&F, filterSize);
    cudaMallocManaged(&O, outputSize);

    initializeInput(I);
    initializeFilter(F);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);
    cudnnSetConvolution2dDescriptor(convDesc, P, P, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);

    int outN, outC, outH, outW;
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &outN, &outC, &outH, &outW);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, outN, outC, outH, outW);

    int requestedAlgoCount = 1;
    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, outputDesc, requestedAlgoCount, &requestedAlgoCount, &algoPerf);
    
    size_t workspaceSize;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algoPerf.algo, &workspaceSize);

    void* workspace;
    cudaMalloc(&workspace, workspaceSize);

    double alpha = 1.0, beta = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    cudnnConvolutionForward(cudnn, &alpha, inputDesc, I, filterDesc, F, convDesc, algoPerf.algo, workspace, workspaceSize, &beta, outputDesc, O);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double checksum = 0.0;
    for (int i = 0; i < K * W * H; ++i) {
        checksum += O[i];
    }
    std::cout << "Checksum (sum of all elements in O): " << checksum << std::endl;

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Kernel execution time: " << elapsed.count() << " seconds" << std::endl;

    cudaFree(I);
    cudaFree(F);
    cudaFree(O);
    cudaFree(workspace);

    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);

    return 0;
}
