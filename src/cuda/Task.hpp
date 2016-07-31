#pragma once

#include "../math/MatrixView.hpp"

namespace cuda {

struct Task {
  math::MatrixView lhs;
  math::MatrixView rhs;
  math::MatrixView out;

  Task(const math::MatrixView &lhs, const math::MatrixView &rhs, const math::MatrixView &out)
      : lhs(lhs), rhs(rhs), out(out) {}
};
}
