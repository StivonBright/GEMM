#include <cstdio>
#include <iostream>
#include <time.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_TILE_SIZE 64
#define BK 16
#define THREAD_TILE_SIZE 8
#define ceil_div(A,B) (((A)+(B)-1)/(B))

using unit_t = std::chrono::nanoseconds;

__global__ void gemm(int M,int N,int K,float *A,float *B, float *C,const float alpha,const float beta){
    
  const int cRow = blockIdx.y * BLOCK_TILE_SIZE;
  const int cCol = blockIdx.x * BLOCK_TILE_SIZE;

  const int threadCol = (threadIdx.x % (BLOCK_TILE_SIZE / THREAD_TILE_SIZE)) * THREAD_TILE_SIZE;
  const int threadRow = (threadIdx.x / (BLOCK_TILE_SIZE / THREAD_TILE_SIZE)) * THREAD_TILE_SIZE;

  // Caching in SMEM for faster access
  __shared__ float tileA[2][BLOCK_TILE_SIZE * BK];
  __shared__ float tileB[2][BLOCK_TILE_SIZE * BK];

  // Advancing pointer to tile start
  A += cRow * K;
  B += cCol;
  C += cRow * N + cCol;

  const int num_threads = (BLOCK_TILE_SIZE/THREAD_TILE_SIZE)*(BLOCK_TILE_SIZE/THREAD_TILE_SIZE);

  // calculating the indices that this thread will load into SMEM
  const int innerRowB = threadIdx.x / (BLOCK_TILE_SIZE/4);
  const int innerColB = threadIdx.x % (BLOCK_TILE_SIZE/4) * 4;
  const int innerRowA = threadIdx.x / (BK/4);
  const int innerColA = threadIdx.x % (BK/4) * 4;

  int strideA = (4 * num_threads) / BK;
  int strideB = (4 * num_threads) / BLOCK_TILE_SIZE;

  float4 tmpA[BK * BLOCK_TILE_SIZE / num_threads / 4];
  float4 tmpB[BK * BLOCK_TILE_SIZE / num_threads / 4];
  
  // allocate thread-level cache for faster access
  float threadResults[THREAD_TILE_SIZE*THREAD_TILE_SIZE] = {0.0};
  float tileAFrag[2][THREAD_TILE_SIZE] = {0.0};
  float tileBFrag[2][THREAD_TILE_SIZE] = {0.0};

  // first load from GMEM to shared memory
#pragma unroll
  for (int offset = 0; offset + strideA <= BLOCK_TILE_SIZE; offset += strideA) {
    float4 tmp = reinterpret_cast<const float4 *>(&A[(innerRowA + offset) * K + innerColA])[0];
    tileA[0][(innerColA + 0) * BLOCK_TILE_SIZE + innerRowA + offset] = tmp.x;
    tileA[0][(innerColA + 1) * BLOCK_TILE_SIZE + innerRowA + offset] = tmp.y;
    tileA[0][(innerColA + 2) * BLOCK_TILE_SIZE + innerRowA + offset] = tmp.z;
    tileA[0][(innerColA + 3) * BLOCK_TILE_SIZE + innerRowA + offset] = tmp.w;
  }
#pragma unroll
  for (int offset = 0; offset + strideB <= BK; offset += strideB) {
    reinterpret_cast<float4 *>(&tileB[0][(innerRowB + offset) * BLOCK_TILE_SIZE + innerColB])[0] = 
    reinterpret_cast<const float4 *>(&B[(innerRowB + offset) * N + innerColB])[0];
  }
  __syncthreads();

  // first shared to register for outer product
#pragma unroll
  for (int i = 0; i < THREAD_TILE_SIZE; i+=4) {
    reinterpret_cast<float4 *>(&tileAFrag[0][i])[0] = reinterpret_cast<float4 *>(&tileA[0][threadRow + i])[0];
    reinterpret_cast<float4 *>(&tileBFrag[0][i])[0] = reinterpret_cast<float4 *>(&tileB[0][threadCol + i])[0];
  }
  int bkIdx = 0;
  int loadIdx = 0;
  int writeIdx = 1;

  do{
    // advance blocktile
    A += BK;    
    B += BK * N;
    bkIdx += BK;

    // no need to execute for last iteration
    if(bkIdx < K){
    // populate the register caches before putting it into shared memory for transposing
    #pragma unroll
      for (int offset = 0; offset + strideA <= BLOCK_TILE_SIZE; offset += strideA) {
        tmpA[offset/strideA] = reinterpret_cast<const float4 *>(&A[(innerRowA + offset) * K + innerColA])[0];
      }
    #pragma unroll
      for (int offset = 0; offset + strideB <= BK; offset += strideB) {
        tmpB[offset/strideB] = reinterpret_cast<const float4 *>(&B[(innerRowB + offset) * N + innerColB])[0];
      }
    }

    // XORing these variables to alternate them
    loadIdx = writeIdx ^ 1;

    // calculate thread tile results - last iteration not included
  #pragma unroll
    for (int dotIdx = 0; dotIdx < BK-1; ++dotIdx) {
      // cache next iteration's tileA and tileB in registers
    #pragma unroll
      for (int i = 0; i < THREAD_TILE_SIZE; ++i) {
        tileAFrag[(dotIdx+1)%2][i] = tileA[loadIdx][(dotIdx+1) * BLOCK_TILE_SIZE + threadRow + i];
        tileBFrag[(dotIdx+1)%2][i] = tileB[loadIdx][(dotIdx+1) * BLOCK_TILE_SIZE + threadCol + i];
      }
      // calculate outer product of currently cached registers
    #pragma unroll
      for (int resIdxM = 0; resIdxM < THREAD_TILE_SIZE; ++resIdxM) {
      #pragma unroll
        for (int resIdxN = 0; resIdxN < THREAD_TILE_SIZE; ++resIdxN) {
          threadResults[resIdxM * THREAD_TILE_SIZE + resIdxN] += tileAFrag[dotIdx%2][resIdxM] * tileBFrag[dotIdx%2][resIdxN];
        }
      }
    }

    // no need to execute for last iteration
    if(bkIdx < K){
    #pragma unroll
      for (int offset = 0; offset + strideA <= BLOCK_TILE_SIZE; offset += strideA) {
        tileA[writeIdx][(innerColA + 0) * BLOCK_TILE_SIZE + innerRowA + offset] = tmpA[offset/strideA].x;
        tileA[writeIdx][(innerColA + 1) * BLOCK_TILE_SIZE + innerRowA + offset] = tmpA[offset/strideA].y;
        tileA[writeIdx][(innerColA + 2) * BLOCK_TILE_SIZE + innerRowA + offset] = tmpA[offset/strideA].z;
        tileA[writeIdx][(innerColA + 3) * BLOCK_TILE_SIZE + innerRowA + offset] = tmpA[offset/strideA].w;
      }
    #pragma unroll
      for (int offset = 0; offset + strideB <= BK; offset += strideB) {
        reinterpret_cast<float4 *>(&tileB[writeIdx][(innerRowB + offset) * BLOCK_TILE_SIZE + innerColB])[0] = 
        reinterpret_cast<float4 *>(&tmpB[offset/strideB])[0];
      }
      __syncthreads();
    #pragma unroll
      for (int i = 0; i < THREAD_TILE_SIZE; i+=4) {
        reinterpret_cast<float4 *>(&tileAFrag[0][i])[0] = reinterpret_cast<float4 *>(&tileA[writeIdx][threadRow + i])[0];
        reinterpret_cast<float4 *>(&tileBFrag[0][i])[0] = reinterpret_cast<float4 *>(&tileB[writeIdx][threadCol + i])[0];
      }
      writeIdx ^= 1;
    }

    // just the last outer product to be calculated here
  #pragma unroll
    for (int resIdxM = 0; resIdxM < THREAD_TILE_SIZE; ++resIdxM) {
    #pragma unroll
      for (int resIdxN = 0; resIdxN < THREAD_TILE_SIZE; ++resIdxN) {
          threadResults[resIdxM * THREAD_TILE_SIZE + resIdxN] += tileAFrag[(BK-1)%2][resIdxM] * tileBFrag[(BK-1)%2][resIdxN];
      }
    }

  }while(bkIdx < K);

// store the results back in GMEM
#pragma unroll
  for (int resIdxM = 0; resIdxM < THREAD_TILE_SIZE; ++resIdxM) {
  #pragma unroll
    for (int resIdxN = 0; resIdxN < THREAD_TILE_SIZE; resIdxN+=4) {
      float4 tmp = reinterpret_cast<float4 *>(&C[(threadRow + resIdxM) * N + threadCol + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * THREAD_TILE_SIZE + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * THREAD_TILE_SIZE + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * THREAD_TILE_SIZE + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * THREAD_TILE_SIZE + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(&C[(threadRow + resIdxM) * N + threadCol + resIdxN])[0] = tmp;
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
    float *A, *B, *C, *C_gemm,*host_A, *host_B, *host_C;
    float alpha= 1.0,beta=0.5;
    float cublas_output[10],gemm_output[10];

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
    cudaMalloc((void **)&C_gemm, M*N*sizeof(float));

    //transfer input from host to device
    cudaMemcpy(A, host_A, sizeof(float) * M * K,cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, sizeof(float) * K * N,cudaMemcpyHostToDevice);
    cudaMemcpy(C, host_C, sizeof(float) * M * N,cudaMemcpyHostToDevice);
    cudaMemcpy(C_gemm, host_C, sizeof(float) * M * N,cudaMemcpyHostToDevice);

    // compare with cuBLAS
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };
    auto cublas_start = std::chrono::steady_clock::now();
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    auto cublas_end = std::chrono::steady_clock::now();
    cudaMemcpy(host_C, C, sizeof(float) * M * N,cudaMemcpyDeviceToHost);
    cudaFree(C);
    free(host_A);
    free(host_B);
    for(int i=0;i<10;++i){
      cublas_output[i] = host_C[5*i+i];
    }

    // create as many blocks as necessary to map all of C
    dim3 gridDim(ceil_div(M,BLOCK_TILE_SIZE), ceil_div(N,BLOCK_TILE_SIZE));
    // 1024 thread per block is maximum
    dim3 blockDim((BLOCK_TILE_SIZE/THREAD_TILE_SIZE)*(BLOCK_TILE_SIZE/THREAD_TILE_SIZE));

    // invoke kernel and get back result
    auto gemm_start = std::chrono::steady_clock::now();
    gemm<<<gridDim, blockDim>>>(M, N, K, A, B, C_gemm, alpha, beta);
    auto gemm_end = std::chrono::steady_clock::now();
    cudaMemcpy(host_C, C_gemm, sizeof(float) * M * N,cudaMemcpyDeviceToHost);
    for(int i=0;i<10;++i){
      gemm_output[i] = host_C[5*i+i];
    }
    //free allocated memory
    cudaFree(A);
    cudaFree(B);  
    cudaFree(C_gemm);
    free(host_C);
    cudaError_t err = cudaGetLastError();  // Error log
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    // checking and validation
    for(int i=0;i<10;++i){
      if(abs(cublas_output[i]-gemm_output[i])>0.01){
        std::cout<<"There is discrepancy in output\n";
      }
    }
    std::cout<<"The efficiency of the kernel is :"<<
    (std::chrono::duration_cast<unit_t>(cublas_end - cublas_start).count())/(std::chrono::duration_cast<unit_t>(gemm_end - gemm_start).count())*100
    <<"%"<<" of cuBLAS\n";

    return 0;
}

