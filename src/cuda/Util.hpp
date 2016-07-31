
#pragma once

#include <cuda_runtime.h>

#define CheckError(ans)                                                                            \
  { cuda::util::OutputError((ans), __FILE__, __LINE__); }

namespace cuda {
namespace util {

void OutputError(cudaError_t code, const char *file, int line);

void *AllocPinned(size_t bufSize);
void FreePinned(void *buf);
}
}
