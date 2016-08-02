
#include "ParallelMatrixMultiply.hpp"
#include <iostream>

int main(int argc, char **argv) {
  ParallelMatrixMultiply taskManager(8);

  std::cout << "hello world" << std::endl;
  return 0;
}
