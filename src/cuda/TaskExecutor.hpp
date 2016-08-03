#pragma once

#include "Task.hpp"
#include <memory>
#include <vector>

namespace cuda {

using StreamId = unsigned int;

class TaskExecutor {
public:
  TaskExecutor();
  virtual ~TaskExecutor();

  StreamId CreateStream(void);

  void ExecuteTask(Task &task, StreamId stream);
  void ExecuteTasks(std::vector<Task> &tasks, StreamId stream);

private:
  struct TaskExecutorImpl;
  std::unique_ptr<TaskExecutorImpl> impl;
};
}
