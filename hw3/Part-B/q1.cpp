#include <iostream>
#include <cstdlib>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <K>" << std::endl;
        return 1;
    }

    int K = std::atoi(argv[1]);
    if (K <= 0) {
        std::cerr << "K must be a positive integer." << std::endl;
        return 1;
    }

    // Total number of elements
    size_t N = K * 1000000;

    // Allocate memory for arrays A, B, and C
    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));

    if (A == nullptr || B == nullptr || C == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        free(A);
        free(B);
        free(C);
        return 1;
    }

    // Initialize arrays A and B with some values
    for (size_t i = 0; i < N; ++i) {
        A[i] = (float) i; 
        B[i] = (float) i;
    }

    // Measure the time taken to add the arrays
    auto start = std::chrono::high_resolution_clock::now();

    // Add elements of A and B into C
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time taken to add " << N << " elements: " << elapsed.count() << " seconds" << std::endl;

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}
