
#pragma once

#include <cstdlib>

namespace cuda {
namespace util {

void *AllocPinned(size_t bufSize);
void FreePinned(void *buf);
}
}
