
#include "TaskExecutor.hpp"
#include <cuda_runtime.h>
#include <utility>

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
    cudaStreamCreate(&newStream);
    streamsMap.emplace_back(nextStreamId++, newStream);
    return streamsMap.back().first;
  }

  void ExecuteTask(const Task &task, StreamId stream) {
    invokeTask(task, stream);
    syncStream(stream);
  }

  void ExecuteTasks(const vector<Task> &tasks, StreamId stream){
    for (const auto &t : tasks) {
      invokeTask(t, stream);
    }
    syncStream(stream);
  }

  void invokeTask(const Task &task, StreamId stream) {

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

void TaskExecutor::ExecuteTask(const Task &task, StreamId stream) {
  impl->ExecuteTask(task, stream);
}

void TaskExecutor::ExecuteTasks(const vector<Task> &tasks, StreamId stream) {
  impl->ExecuteTasks(tasks, stream);
}
