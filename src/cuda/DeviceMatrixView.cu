
#include "DeviceMatrixView.hpp"
#include "Util.cuh"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

DeviceMatrixView DeviceMatrixView::New(int rows, int cols) {
  assert(rows > 0 && cols > 0);

  DeviceMatrixView result;
  result.rows = rows;
  result.cols = cols;

  size_t width = cols * sizeof(float);
  size_t height = rows;

  cudaError_t err = cudaMallocPitch(&result.data, &result.pitch, width, height);
  CheckError(err);

  return result;
}

void DeviceMatrixView::Delete(DeviceMatrixView &dmv) {
  cudaError_t err = cudaFree(dmv.data);
  CheckError(err);
  dmv.data = nullptr;
}
