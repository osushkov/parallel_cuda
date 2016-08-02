
#include "AsyncTask.hpp"
#include "ParallelMatrixMultiply.hpp"
#include "common/Common.hpp"
#include "math/Math.hpp"
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <utility>

EMatrix createMatrix(void) {
  static constexpr unsigned SIZE = 1000;

  EMatrix result(SIZE, SIZE);

  for (int r = 0; r < result.rows(); r++) {
    for (int c = 0; c < result.cols(); c++) {
      result(r, c) = math::UnitRand();
    }
  }

  return result;
}

int main(int argc, char **argv) {
  ParallelMatrixMultiply taskManager(4);

  vector<EMatrix> matrices;
  matrices.reserve(100);

  cout << "generting matrices" << endl;
  for (unsigned i = 0; i < 100; i++) {
    matrices.push_back(createMatrix());
  }
  cout << "done generting matrices" << endl;

  unsigned numResults = 0;
  mutex m;
  condition_variable cv;

  // Serial version
  // for (unsigned i = 0; i < 1000; i++) {
  //   EMatrix r = matrices[rand() % matrices.size()] * matrices[rand() % matrices.size()];
  //   numResults++;
  // }

  // Parallel
  for (unsigned i = 0; i < 1000; i++) {
    AsyncTask task(matrices[rand() % matrices.size()], matrices[rand() % matrices.size()],
                   [&](const EMatrix &result) {
                     std::unique_lock<std::mutex> lk(m);
                     numResults++;
                     cv.notify_one();
                   });

    taskManager.PushTask(task);
  }

  cout << "finished pushing tasks" << std::endl;

  while (true) {
    std::unique_lock<std::mutex> lk(m);
    if (numResults >= 1000) {
      break;
    }

    cv.wait(lk, [&numResults]() { return numResults >= 1000; });
  }

  return 0;
}
