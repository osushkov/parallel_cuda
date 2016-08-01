#pragma once

#include "AsyncTask.hpp"
#include "common/Common.hpp"

class ParallelMatrixMultiply {
public:
  ParallelMatrixMultiply(unsigned numStreams);
  virtual ~ParallelMatrixMultiply();

  void PushTask(const AsyncTask &task);

private:
  struct ParallelMatrixMultiplyImpl;
  uptr<ParallelMatrixMultiplyImpl> impl;
};
