/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "apex_raja.hpp"

// Statically loading plugin.
static RAJA::util::PluginRegistry::add<ApexPlugin> P("APEX", "APEX - Autonomic Performance Environment for eXascale");

// Dynamically loading plugin.
extern "C" RAJA::util::PluginStrategy *getPlugin ()
{
  return new ApexPlugin;
}