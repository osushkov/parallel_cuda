
#include "ParallelMatrixMultiply.hpp"
#include "Constants.hpp"
#include "cuda/DeviceMatrixView.hpp"
#include "cuda/TaskExecutor.hpp"
#include "cuda/Util.hpp"
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

struct ExecutorContext {
  cuda::StreamId streamId;

  math::MatrixView pinnedBuf0;
  math::MatrixView pinnedBuf1;
  math::MatrixView pinnedBuf2;

  DeviceMatrixView deviceMatrix0;
  DeviceMatrixView deviceMatrix1;
  DeviceMatrixView deviceMatrix2;
};

struct ParallelMatrixMultiply::ParallelMatrixMultiplyImpl {
  bool shouldStop;
  mutex m; // controls access to tasks for the workers.
  condition_variable cv;

  vector<AsyncTask> tasks;
  vector<thread> workers;

  cuda::TaskExecutor taskExecutor;

  ParallelMatrixMultiplyImpl(unsigned numStreams) : shouldStop(false) {
    assert(numStreams > 0);
    createWorkers(numStreams);
  }

  ~ParallelMatrixMultiplyImpl() { stopWorkers(); }

  void PushTask(const AsyncTask &task) {
    std::unique_lock<std::mutex> lk(m);
    tasks.push_back(task);

    cv.notify_one();
  }

  void createWorkers(unsigned numWorkers) {
    for (unsigned i = 0; i < numWorkers; i++) {
      workers.emplace_back([this, i] {
        ExecutorContext ctx = createContext();

        while (true) {
          std::unique_lock<std::mutex> lk(m);

          if (tasks.empty() && !shouldStop) {
            cv.wait(lk, [this]() { return !tasks.empty() || shouldStop; });
          }

          if (shouldStop) {
            break;
          } else {
            assert(!tasks.empty());

            AsyncTask task = tasks.back();
            tasks.pop_back();

            lk.unlock();

            doTask(task, ctx);
          }
        }

        cleanup(ctx);
      });
    }
  }

  void doTask(AsyncTask &task, ExecutorContext context) {
    memcpy(context.pinnedBuf0.data, task.lhs.data(),
           task.lhs.rows() * task.lhs.cols() * sizeof(float));

    memcpy(context.pinnedBuf1.data, task.rhs.data(),
           task.rhs.rows() * task.rhs.cols() * sizeof(float));

    cuda::Task task0 = cuda::Task::HDCopy(context.pinnedBuf0, context.deviceMatrix0);
    cuda::Task task1 = cuda::Task::HDCopy(context.pinnedBuf1, context.deviceMatrix1);
    cuda::Task task2 =
        cuda::Task::Mult(context.deviceMatrix0, context.deviceMatrix1, context.deviceMatrix2);
    cuda::Task task3 = cuda::Task::DHCopy(context.deviceMatrix2, context.pinnedBuf2);

    vector<cuda::Task> tasks{task0, task1, task2, task3};
    taskExecutor.ExecuteTasks(tasks, context.streamId);

    EMatrix result(task.lhs.rows(), task.rhs.cols());
    math::MatrixView rv = math::GetMatrixView(result);

    memcpy(rv.data, context.pinnedBuf2.data, rv.rows * rv.cols * sizeof(float));

    task.resultCallback(result);
  }

  ExecutorContext createContext(void) {
    ExecutorContext context;

    context.streamId = taskExecutor.CreateStream();
    context.deviceMatrix0 = DeviceMatrixView::New(MATRIX_SIZE, MATRIX_SIZE);
    context.deviceMatrix1 = DeviceMatrixView::New(MATRIX_SIZE, MATRIX_SIZE);
    context.deviceMatrix2 = DeviceMatrixView::New(MATRIX_SIZE, MATRIX_SIZE);

    context.pinnedBuf0 = createPinnedView();
    context.pinnedBuf1 = createPinnedView();
    context.pinnedBuf2 = createPinnedView();

    return context;
  }

  math::MatrixView createPinnedView(void) {
    math::MatrixView result;
    result.rows = MATRIX_SIZE;
    result.cols = MATRIX_SIZE;
    result.data = (float *)cuda::util::AllocPinned(result.rows * result.cols * sizeof(float));
    return result;
  }

  void cleanup(ExecutorContext context) {
    DeviceMatrixView::Delete(context.deviceMatrix0);
    DeviceMatrixView::Delete(context.deviceMatrix1);
    DeviceMatrixView::Delete(context.deviceMatrix2);

    cuda::util::FreePinned(context.pinnedBuf0.data);
    cuda::util::FreePinned(context.pinnedBuf1.data);
    cuda::util::FreePinned(context.pinnedBuf2.data);
  }

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
