#include "matrix_operations.h"
#include <iostream>
#include <vector>
#include <cmath>

// Incluye las cabeceras de CUDA y cuBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h> // Para cudaMalloc, cudaMemcpy, cudaFree, cudaGetLastError, etc.

using namespace std;

// --- Funciones de Ayuda para Manejo de Errores CUDA/cuBLAS ---
// Puedes poner estas funciones en un archivo de utilidad aparte si quieres, pero por simplicidad las incluimos aquí.

// Función para chequear errores de CUDA Runtime API
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << err << "(" << func << ") : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Función para chequear errores de cuBLAS API
#define CHECK_CUBLAS_ERROR(val) check_cublas((val), #val, __FILE__, __LINE__)
void check_cublas(cublasStatus_t status, const char* const func, const char* const file, const int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        // cublasGetStatusString no existe en la mayoría de las versiones de cuBLAS, así que imprimimos el enum
        std::cerr << "CUBLAS Error at " << file << ":" << line << " code=" << status << "(" << func << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Implementación de funciones existentes (sin cambios)
void addMatrices(const vector<vector<float> >& a, const vector<vector<float> >& b, vector<vector<float> >& c) {
    int m = a.size();
    int n = a[0].size();
    if (m != b.size() || n != b[0].size()) {
        throw invalid_argument("Matrices of sizes " + to_string(m) + "x" + to_string(n) + " and " + to_string(b.size()) + "x" + to_string(b[0].size()) + " are not compatible for addition");
    }
    c.resize(m, vector<float>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}
//
//void multiplyMatrices(const vector<vector<float> >& a, const vector<vector<float> >& b, vector<vector<float> >& c) {
//    int m = a.size();
//    if (m == 0) throw invalid_argument("Matrix 'a' is empty");
//    int n = a[0].size();
//    if (n == 0) throw invalid_argument("Matrix 'a' has no columns");
//    int b_rows = b.size();
//    if (b_rows == 0) throw invalid_argument("Matrix 'b' is empty");
//    int p = b[0].size();
//    if (p == 0) throw invalid_argument("Matrix 'b' has no columns");
//    if (n != b_rows) {
//        throw invalid_argument("Matrices of sizes " + to_string(m) + "x" + to_string(n) + " and " + to_string(b_rows) + "x" + to_string(p) + " are not compatible for multiplication");
//    }
//    c.resize(m, vector<float>(p, 0.0f));
//    for (int i = 0; i < m; i++) {
//        for (int j = 0; j < p; j++) {
//            for (int k = 0; k < n; k++) {
//                c[i][j] += a[i][k] * b[k][j];
//            }
//        }
//    }
//}

void broadcastMultiply(const vector<float>& a, const vector<float>& b, vector<vector<float> >& c) {
    int m = a.size();
    int n = b.size();
    c.resize(m, vector<float>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i] * b[j];
        }
    }
}

void print2DMatrix(const vector<vector<float> >& a) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
}

// --- Nueva implementación de multiplicación de matrices usando cuBLAS ---
void multiplyMatrices(const vector<vector<float>>& a, const vector<vector<float>>& b, vector<vector<float>>& c) {
    // Obtener dimensiones de las matrices
    int M = a.size();    // Filas de A
    if (M == 0) throw invalid_argument("Matrix 'a' is empty.");
    int N = a[0].size(); // Columnas de A y Filas de B
    if (N == 0) throw invalid_argument("Matrix 'a' has no columns.");
    int K = b[0].size(); // Columnas de B
    if (b.size() != N) {
        throw invalid_argument("Matrices are not compatible for multiplication. A cols (" +
            to_string(N) + ") != B rows (" + to_string(b.size()) + ")");
    }
    if (K == 0) throw invalid_argument("Matrix 'b' has no columns.");

    c.resize(M, vector<float>(K)); // Redimensionar la matriz de resultado C

    // Aplanar matrices a formato 1D (row-major)
    vector<float> flat_a(M * N);
    vector<float> flat_b(N * K);
    vector<float> flat_c(M * K);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            flat_a[i * N + j] = a[i][j];

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            flat_b[i * K + j] = b[i][j];

    // Punteros a memoria en la GPU
    float* d_a, * d_b, * d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, M * K * sizeof(float)));

    // Copiar datos a la GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, flat_a.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, flat_b.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

    // Crear handle de cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    // Configurar parámetros para multiplicación
    float alpha = 1.0f;
    float beta = 0.0f;

    // Versión CORRECTA para matrices row-major:
    // C = A * B (row-major) es equivalente a C^T = B^T * A^T (column-major)
    // Por lo tanto usamos:
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, d_b, K, d_a, N, &beta, d_c, K);

    CHECK_CUBLAS_ERROR(cublasSgemm(handle,
        CUBLAS_OP_N,  // No transponer B (porque ya está en el orden correcto para B^T)
        CUBLAS_OP_N,  // No transponer A (porque ya está en el orden correcto para A^T)
        K,            // Columnas de B (filas de B^T)
        M,            // Filas de A (columnas de A^T)
        N,            // Dimensión común
        &alpha,
        d_b,          // Matriz B
        K,            // Leading dimension de B (columnas en B row-major)
        d_a,          // Matriz A
        N,            // Leading dimension de A (columnas en A row-major)
        &beta,
        d_c,          // Matriz resultado C
        K));          // Leading dimension de C (columnas en C row-major)

    // Copiar resultado de vuelta a CPU
    CHECK_CUDA_ERROR(cudaMemcpy(flat_c.data(), d_c, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    // Reconstruir matriz 2D
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            c[i][j] = flat_c[i * K + j];

    // Liberar recursos
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
}