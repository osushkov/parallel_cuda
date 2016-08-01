
#include "ParallelMatrixMultiply.hpp"
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <thread>

struct ParallelMatrixMultiply::ParallelMatrixMultiplyImpl {
  bool shouldStop;
  mutex m; // controls access to tasks for the workers.
  condition_variable cv;

  vector<AsyncTask> tasks;
  vector<thread> workers;

  ParallelMatrixMultiplyImpl(unsigned numStreams) : shouldStop(false) {
    assert(numStreams > 0);
    createWorkers(numStreams);
  }

  ~ParallelMatrixMultiplyImpl() { stopWorkers(); }

  void PushTask(const AsyncTask &task) {}

  void createWorkers(unsigned numWorkers) {
    for (unsigned i = 0; i < numWorkers; i++) {
      workers.emplace_back([this] {
        while (true) {
          std::unique_lock<std::mutex> lk(m);
          cv.wait(lk, [this]() { return !tasks.empty() || shouldStop; });

          if (shouldStop) {
            break;
          } else {
            assert(!tasks.empty());

            AsyncTask task = tasks.back();
            tasks.pop_back();

            lk.release();
            doTask(task);
          }
        }
      });
    }
  }

  void doTask(const AsyncTask &task) {}

  void stopWorkers(void) {
    {
      std::lock_guard<std::mutex> l(m);
      shouldStop = true;
    }

    cv.notify_all();
    for (auto &w : workers) {
      w.join();
    }
  }
};

ParallelMatrixMultiply::ParallelMatrixMultiply(unsigned numStreams)
    : impl(new ParallelMatrixMultiplyImpl(numStreams)) {
  assert(numStreams > 0);
}

ParallelMatrixMultiply::~ParallelMatrixMultiply() = default;

void ParallelMatrixMultiply::PushTask(const AsyncTask &task) { impl->PushTask(task); }
