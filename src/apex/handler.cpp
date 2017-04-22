#include "handler.hpp"

#if defined(_MSC_VER)
namespace apex {
  std::chrono::microseconds handler::default_period(100000);
}
#endif
