#pragma once

#include "Util.hpp"
#include <cassert>
#include <cuda_runtime.h>

struct DeviceMatrixView {
  int rows;
  int cols;

  size_t pitch;
  float *data;

  __device__ float *Elem(unsigned r, unsigned c) {
    assert(r < rows && c < cols);
    return (float *)((char *)data + r * pitch) + c;
  }

  static DeviceMatrixView New(int rows, int cols) {
    assert(rows > 0 && cols > 0);

    DeviceMatrixView result;
    result.rows = rows;
    result.cols = cols;

    cudaError_t err = cudaMallocPitch(&result.data, &result.pitch, cols, rows);
    CheckError(err);

    return result;
  }

  static void Delete(DeviceMatrixView &dmv) {
    cudaError_t err = cudaFree(dmv.data);
    CheckError(err);
    dmv.data = nullptr;
  }
};
