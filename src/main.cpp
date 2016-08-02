
#include "common/Common.hpp"
#include "math/Math.hpp"
#include "AsyncTask.hpp"
#include "ParallelMatrixMultiply.hpp"
#include <iostream>
#include <utility>
#include <cstdlib>


pair<EMatrix, EMatrix> createMM(void) {
  static constexpr unsigned SIZE = 3000;

  EMatrix lhs(SIZE, SIZE);
  EMatrix rhs(SIZE, SIZE);

  for (int r = 0; r < lhs.rows(); r++) {
    for (int c = 0; c < lhs.cols(); c++) {
      lhs(r, c) = math::UnitRand();
      rhs(r, c) = math::UnitRand();
    }
  }

  return make_pair(lhs, rhs);
}

int main(int argc, char **argv) {
  ParallelMatrixMultiply taskManager(8);

  for (unsigned i = 0; i < 1000; i++) {
    auto data = createMM();
    AsyncTask task(data.first, data.second, [](const EMatrix &result) {
      // cout << "got result" << endl;
    });

    taskManager.PushTask(task);
  }

  cout << "hello world" << std::endl;
  sleep(5);
  return 0;
}
