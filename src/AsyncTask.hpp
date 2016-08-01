#pragma once

#include "math/Math.hpp"
#include <functional>

struct AsyncTask {
  EMatrix lhs;
  EMatrix rhs;

  std::function<void(const EMatrix &)> resultCallback;

  AsyncTask(const EMatrix &lhs, const EMatrix &rhs, std::function<void(const EMatrix &)> cb)
      : lhs(lhs), rhs(rhs), resultCallback(cb) {}
};
