#pragma once

#include <cstdlib>

struct DeviceMatrixView {
  int rows;
  int cols;

  size_t pitch;
  float *data;

  static DeviceMatrixView New(int rows, int cols);
  static void Delete(DeviceMatrixView &dmv);
};
