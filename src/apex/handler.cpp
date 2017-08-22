#include "handler.hpp"

#if defined(_MSC_VER) || defined(__APPLE__)
namespace apex {
  std::chrono::microseconds handler::default_period(100000);
}
#endif
