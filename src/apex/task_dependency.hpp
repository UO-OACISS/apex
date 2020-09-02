/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include "task_identifier.hpp"

namespace apex {

class task_dependency {
public:
  task_identifier parent;
  task_identifier child;
  task_dependency(task_identifier * p, task_identifier * c) :
    parent(*p), child(*c) {};
  ~task_dependency() { }
};

}

