
#include "CudaKernel.hpp"
#include "DeviceMatrixView.hpp"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cstdio>

using namespace std;

// Threads per block in X and Y dimensions.
static constexpr int tpbX = 32;
static constexpr int tpbY = 32;

__device__ float *Elem(DeviceMatrixView v, unsigned r, unsigned c) {
  assert(r < v.rows && c < v.cols);
  return (float *)((char *)v.data + r * v.pitch) + c;
}

__global__ void multKernel(DeviceMatrixView lhs, DeviceMatrixView rhs, DeviceMatrixView out,
                           unsigned spitch) {
  extern __shared__ float buf[]; // shared memory buffer

  const int outRow = blockDim.y * blockIdx.y + threadIdx.y;
  const int outCol = blockDim.x * blockIdx.x + threadIdx.x;

  const int numChunks = (lhs.cols + blockDim.x - 1) / blockDim.x;

  // buffer for holding the lhs matrix chunk
  float *lhsChunk = (float *) buf;

  // buffer for holding the prev outputs matrix chunk
  float *rhsChunk = (float *) &buf[spitch * blockDim.y];

  float value = 0.0f;
  for (int i = 0; i < numChunks; i++) {
    const int chunkOffset = i * blockDim.x;
    const int chunkIndex = threadIdx.x + threadIdx.y * blockDim.x;

    const int lhsRow = outRow;
    const int lhsCol = chunkOffset + threadIdx.x;

    const int rhsRow = chunkOffset + threadIdx.y;
    const int rhsCol = outCol;

    if (lhsRow < lhs.rows && lhsCol < lhs.cols) {
      lhsChunk[chunkIndex] = *Elem(lhs, lhsRow, lhsCol);
    }
    if (rhsRow < rhs.rows && rhsCol < rhs.cols) {
      rhsChunk[chunkIndex] = *Elem(rhs, rhsRow, rhsCol);
    }

    __syncthreads();

    if (outRow < out.rows && outCol < out.cols) {
      int chunkLim = min(blockDim.x, lhs.cols - chunkOffset);
      for (int j = 0; j < chunkLim; j++) {
        value += lhsChunk[j + threadIdx.y * blockDim.x] * rhsChunk[threadIdx.x + j * blockDim.x];
      }
    }
    __syncthreads();
  }

  if (outRow < out.rows && outCol < out.cols) {
    float *outElem = Elem(out, outRow, outCol);
    *outElem = value;
  }
}

void CudaKernel::Multiply(const DeviceMatrixView &lhs, const DeviceMatrixView &rhs,
                          DeviceMatrixView &out, cudaStream_t stream) {

  // Blocks per grid in X and Y dimensions.
  int bpgX = (out.cols + tpbX - 1) / tpbX;
  int bpgY = (out.rows + tpbY - 1) / tpbY;

  unsigned spitch = (tpbX + 1);
  size_t sharedMemSize = 2 * spitch * tpbY * sizeof(float);

  multKernel<<<dim3(bpgX, bpgY, 1), dim3(tpbX, tpbY, 1), sharedMemSize, stream>>>(
    lhs, rhs, out, spitch);
}
