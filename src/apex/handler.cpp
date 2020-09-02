/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "handler.hpp"

#if defined(_MSC_VER) || defined(__APPLE__)
namespace apex {
  std::chrono::microseconds handler::default_period(100000);
}
#endif
