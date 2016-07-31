
#pragma once

#include "DeviceMatrixView.hpp"
#include <vector>

namespace CudaKernel {

void Multiply(const DeviceMatrixView &lhs, const DeviceMatrixView &rhs, DeviceMatrixView &out,
              cudaStream_t stream);
}
