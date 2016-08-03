#pragma once

#include "../math/MatrixView.hpp"
#include "DeviceMatrixView.hpp"

namespace cuda {

enum class TaskType {
  DH_COPY,
  HD_COPY,
  MULT,
};

struct DHCopyTask {
  DeviceMatrixView src;
  math::MatrixView dst;

  DHCopyTask() = default;
  DHCopyTask(DeviceMatrixView src, math::MatrixView dst) : src(src), dst(dst) {}
};

struct HDCopyTask {
  math::MatrixView src;
  DeviceMatrixView dst;

  HDCopyTask() = default;
  HDCopyTask(math::MatrixView src, DeviceMatrixView dst) : src(src), dst(dst) {}
};

struct MultTask {
  DeviceMatrixView lhs;
  DeviceMatrixView rhs;
  DeviceMatrixView out;

  MultTask() = default;
  MultTask(DeviceMatrixView lhs, DeviceMatrixView rhs, DeviceMatrixView out)
      : lhs(lhs), rhs(rhs), out(out) {}
};

struct Task {
  TaskType type;

  DHCopyTask dhCopy;
  HDCopyTask hdCopy;
  MultTask mult;

  static Task DHCopy(DeviceMatrixView src, math::MatrixView dst) {
    Task result;
    result.type = TaskType::DH_COPY;
    result.dhCopy = DHCopyTask(src, dst);
    return result;
  }

  static Task HDCopy(math::MatrixView src, DeviceMatrixView dst) {
    Task result;
    result.type = TaskType::HD_COPY;
    result.hdCopy = HDCopyTask(src, dst);
    return result;
  }

  static Task Mult(DeviceMatrixView lhs, DeviceMatrixView rhs, DeviceMatrixView out) {
    Task result;
    result.type = TaskType::MULT;
    result.mult = MultTask(lhs, rhs, out);
    return result;
  }
};
}
