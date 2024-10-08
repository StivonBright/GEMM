#include <cstdio>
#include <iostream>
#include <time.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_TILE_SIZE 32
#define THREAD_TILE_SIZE 4
#define ceil_div(A,B) (A+B-1)/B

__global__ void gemm(int M,int N,int K,const float *A,const float *B, float *C,const float alpha,const float beta){
    
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int threadCol = threadIdx.x % (BLOCK_TILE_SIZE / THREAD_TILE_SIZE);
  const int threadRow = threadIdx.x / (BLOCK_TILE_SIZE / THREAD_TILE_SIZE);

  // Caching in SMEM for faster access
  __shared__ float tileA[BLOCK_TILE_SIZE * BLOCK_TILE_SIZE];
  __shared__ float tileB[BLOCK_TILE_SIZE * BLOCK_TILE_SIZE];

  // Advancing pointer to tile start
  A += cRow * BLOCK_TILE_SIZE * K;
  B += cCol * BLOCK_TILE_SIZE;
  C += cRow * BLOCK_TILE_SIZE * N + cCol * BLOCK_TILE_SIZE;

  // calculating the indices that this thread will load into SMEM
  const int innerRow = threadIdx.x / BLOCK_TILE_SIZE;
  const int innerCol = threadIdx.x % BLOCK_TILE_SIZE;
  
  // allocate thread-level cache for faster access
  float threadResults[THREAD_TILE_SIZE] = {0.0};
  float tileAReg[THREAD_TILE_SIZE] = {0.0};
  float tileBReg[THREAD_TILE_SIZE] = {0.0};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_TILE_SIZE) {
    // populate the SMEM caches
    for (int offset = 0; offset < BLOCK_TILE_SIZE; offset += THREAD_TILE_SIZE) {
      tileA[(innerRow + offset) * BLOCK_TILE_SIZE + innerCol] = A[(innerRow + offset) * K + innerCol];
      tileB[(innerRow + offset) * BLOCK_TILE_SIZE + innerCol] = B[(innerRow + offset) * N + innerCol];
    }
    // make sure SMEM is populated before proceeding
    __syncthreads();

    // advance blocktile
    A += BLOCK_TILE_SIZE;    
    B += BLOCK_TILE_SIZE * N; 

    // calculate thread tile results - dot product of tiles along A and B
    for (int dotIdx = 0; dotIdx < BLOCK_TILE_SIZE; ++dotIdx) {
      // cache tileA and tileB in registers
      for (int i = 0; i < THREAD_TILE_SIZE; ++i) {
        tileAReg[i] = tileA[(threadRow * THREAD_TILE_SIZE + i) * BLOCK_TILE_SIZE + dotIdx];
      }
      for (int i = 0; i < THREAD_TILE_SIZE; ++i) {
        tileBReg[i] = tileB[dotIdx * BLOCK_TILE_SIZE + threadCol * THREAD_TILE_SIZE + i];
      }
      for (int resIdxM = 0; resIdxM < THREAD_TILE_SIZE; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < THREAD_TILE_SIZE; ++resIdxN) {
          threadResults[resIdxM * THREAD_TILE_SIZE + resIdxN] += tileAReg[resIdxM] * tileBReg[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // store the results back in GMEM
  for (int resIdxM = 0; resIdxM < THREAD_TILE_SIZE; ++resIdxM) {
    for (int resIdxN = 0; resIdxN < THREAD_TILE_SIZE; ++resIdxN) {
      C[(threadRow * THREAD_TILE_SIZE + resIdxM) * N + threadCol * THREAD_TILE_SIZE + resIdxN] = alpha * threadResults[resIdxM * THREAD_TILE_SIZE + resIdxN] 
      + beta * C[(threadRow * THREAD_TILE_SIZE + resIdxM) * N + threadCol * THREAD_TILE_SIZE + resIdxN];
    }
  }

}

void randomize_matrix(float *mat, int N) {
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

int main(){
    // dimensions for test data matrix
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    // allocating uninitialized matrices
    float *A, *B, *C,*host_A, *host_B, *host_C;
    float alpha= 1.0,beta=0.8;

    // random input generation on host side
    host_A = (float *)malloc(M*K*sizeof(float));
    host_B = (float *)malloc(K*N*sizeof(float));
    host_C = (float *)malloc(M*N*sizeof(float));

    randomize_matrix(host_A, M*K);
    randomize_matrix(host_B, K*N);
    randomize_matrix(host_C, M*N);

    //allocate device mem
    cudaMalloc((void **)&A, M*K*sizeof(float));
    cudaMalloc((void **)&B, K*N*sizeof(float));
    cudaMalloc((void **)&C, M*N*sizeof(float));

    //transfer input from host to device
    cudaMemcpy(A, host_A, sizeof(float) * M * K,cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, sizeof(float) * K * N,cudaMemcpyHostToDevice);
    cudaMemcpy(C, host_C, sizeof(float) * M * N,cudaMemcpyHostToDevice);
    // create as many blocks as necessary to map all of C
    dim3 gridDim(ceil_div(M,BLOCK_TILE_SIZE), ceil_div(N,BLOCK_TILE_SIZE));
    // 1024 thread per block is maximum
    dim3 blockDim(8 * 8);
    // invoke kernel and get back result
    gemm<<<gridDim, blockDim>>>(M, N, N, A, B, C, alpha, beta);
    cudaMemcpy(host_C, C, sizeof(float) * M * N,cudaMemcpyDeviceToHost);
    // compare with cuBLAS
    /*cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, host_B, CUDA_R_32F,
               N, host_A, CUDA_R_32F, K, &beta, host_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    */
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(host_A);
    free(host_B);
    free(host_C);
    cudaError_t err = cudaGetLastError();  // Error log
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    
    return 0;
}

