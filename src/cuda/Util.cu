
#include "Util.hpp"
#include <iostream>
#include <cassert>

using namespace cuda;

void util::OutputError(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "GPU error: " << cudaGetErrorString(code) << " "
        << file << "(" << line << ")" << std::endl;
    exit(code);
  }
}

void *util::AllocPinned(size_t bufSize) {
  void* result = nullptr;

  cudaError_t err = cudaHostAlloc(&result, bufSize, cudaHostAllocPortable);
  CheckError(err);
  assert(result != nullptr);

  return result;
}

void util::FreePinned(void *buf) {
  assert(buf != nullptr);
  cudaError_t err = cudaFreeHost(buf);
  CheckError(err);
}
