#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        printf("cuDNN error: %s\n", cudnnGetErrorString(status)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUDA(call) do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(status)); \
        exit(1); \
    } \
} while(0)

// Problem dimensions
#define H 1024
#define W 1024
#define C 3
#define FH 3
#define FW 3
#define K 64
#define P 1

int main() {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Allocate host memory
    double *h_I = (double*)malloc(C * H * W * sizeof(double));
    double *h_F = (double*)malloc(K * C * FH * FW * sizeof(double));
    double *h_O = (double*)malloc(K * H * W * sizeof(double));
    
    // Initialize input tensor I
    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                h_I[c*H*W + x*W + y] = c * (x + y);
            }
        }
    }
    
    // Initialize filters F
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++) {
                for (int j = 0; j < FW; j++) {
                    h_F[k*C*FH*FW + c*FH*FW + i*FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    // Allocate device memory
    double *d_I, *d_F, *d_O;
    CHECK_CUDA(cudaMalloc(&d_I, C * H * W * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_F, K * C * FH * FW * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_O, K * H * W * sizeof(double)));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_I, h_I, C * H * W * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_F, h_F, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice));

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    // Set tensor descriptors
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NCHW,    // Format
        CUDNN_DATA_DOUBLE,    // Data type
        1,                    // Batch size
        C,                    // Channels
        H,                    // Height
        W                     // Width
    ));

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_descriptor,
        CUDNN_DATA_DOUBLE,    // Data type
        CUDNN_TENSOR_NCHW,    // Format
        K,                    // Number of output feature maps
        C,                    // Number of input feature maps
        FH,                   // Filter height
        FW                    // Filter width
    ));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        convolution_descriptor,
        P, P,                 // Zero-padding height and width
        1, 1,                 // Vertical and horizontal stride
        1, 1,                 // Vertical and horizontal dilation
        CUDNN_CONVOLUTION,    // Mode
        CUDNN_DATA_DOUBLE    // Compute type
    ));

    // Get output dimensions
    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        convolution_descriptor,
        input_descriptor,
        filter_descriptor,
        &out_n,
        &out_c,
        &out_h,
        &out_w
    ));

    // Set output descriptor
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_descriptor,
        CUDNN_TENSOR_NCHW,    // Format
        CUDNN_DATA_DOUBLE,    // Data type
        1,                    // Batch size
        K,                    // Channels
        out_h,               // Height
        out_w                // Width
    ));

    // Choose the best algorithm
    cudnnConvolutionFwdAlgo_t algorithm;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        input_descriptor,
        filter_descriptor,
        convolution_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,  // No memory limit
        &algorithm
    ));

    // Get workspace size and allocate
    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        filter_descriptor,
        convolution_descriptor,
        output_descriptor,
        algorithm,
        &workspace_size
    ));

    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Perform convolution
    const double alpha = 1.0;
    const double beta = 0.0;

    // Record start time
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor, d_I,
        filter_descriptor, d_F,
        convolution_descriptor,
        algorithm,
        d_workspace, workspace_size,
        &beta,
        output_descriptor, d_O
    ));

    // Record stop time
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_O, d_O, K * H * W * sizeof(double), cudaMemcpyDeviceToHost));

    // Calculate checksum
    double checksum = 0.0;
    for (int k = 0; k < K; k++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                checksum += h_O[k*H*W + x*W + y];
            }
        }
    }
    printf("Checksum: %.6e\n", checksum);

    // Cleanup
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_I));
    CHECK_CUDA(cudaFree(d_F));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_descriptor));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
    CHECK_CUDNN(cudnnDestroy(cudnn));
    
    free(h_I);
    free(h_F);
    free(h_O);

    return 0;
}