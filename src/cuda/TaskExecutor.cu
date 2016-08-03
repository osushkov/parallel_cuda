
#include "TaskExecutor.hpp"
#include "Util.cuh"
#include "CudaKernel.hpp"
#include <cuda_runtime.h>
#include <utility>
#include <iostream>

using namespace cuda;
using namespace std;

struct TaskExecutor::TaskExecutorImpl {
  StreamId nextStreamId;
  vector<pair<StreamId, cudaStream_t>> streamsMap;

  TaskExecutorImpl() : nextStreamId(1) {};

  ~TaskExecutorImpl() {
    for (auto &s : streamsMap) {
      cudaStreamDestroy(s.second);
    }
  }

  StreamId CreateStream(void) {
    cudaStream_t newStream;
    cudaStreamCreateWithFlags(&newStream, cudaStreamNonBlocking);
    streamsMap.emplace_back(nextStreamId++, newStream);
    return streamsMap.back().first;
  }

  void ExecuteTask(Task &task, StreamId stream) {
    invokeTask(task, stream);
    syncStream(stream);
  }

  void ExecuteTasks(vector<Task> &tasks, StreamId stream){
    for (auto &t : tasks) {
      invokeTask(t, stream);
    }
    syncStream(stream);
  }

  void invokeTask(Task &task, StreamId stream) {
    if (task.type == TaskType::DH_COPY) {
      cudaError_t err = cudaMemcpy2DAsync(
        task.dhCopy.dst.data, task.dhCopy.dst.cols * sizeof(float),
        task.dhCopy.src.data, task.dhCopy.src.pitch,
        task.dhCopy.src.cols * sizeof(float), task.dhCopy.src.rows,
        cudaMemcpyDeviceToHost, getStream(stream));

      CheckError(err);
    } else if (task.type == TaskType::HD_COPY) {
      cudaError_t err = cudaMemcpy2DAsync(
        task.hdCopy.dst.data, task.hdCopy.dst.pitch,
        task.hdCopy.src.data, task.hdCopy.src.cols * sizeof(float),
        task.hdCopy.src.cols * sizeof(float), task.hdCopy.src.rows,
        cudaMemcpyHostToDevice, getStream(stream));

      CheckError(err);

    } else if (task.type == TaskType::MULT) {
      CudaKernel::Multiply(task.mult.lhs, task.mult.rhs, task.mult.out, getStream(stream));
    } else {
      assert(false);
    }
  }

  void syncStream(StreamId stream) {
    cudaStreamSynchronize(getStream(stream));
  }

  cudaStream_t getStream(StreamId id) {
    assert(!streamsMap.empty());
    for (const auto &e : streamsMap) {
      if (e.first == id) {
        return e.second;
      }
    }
    assert(false);
    return 0;
  }
};

TaskExecutor::TaskExecutor() : impl(new TaskExecutorImpl()) {}
TaskExecutor::~TaskExecutor() = default;

StreamId TaskExecutor::CreateStream(void) { return impl->CreateStream(); }

void TaskExecutor::ExecuteTask(Task &task, StreamId stream) {
  impl->ExecuteTask(task, stream);
}

void TaskExecutor::ExecuteTasks(vector<Task> &tasks, StreamId stream) {
  impl->ExecuteTasks(tasks, stream);
}
