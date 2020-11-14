#include "apex_raja.hpp"

// Statically loading plugin.
static RAJA::util::PluginRegistry::add<ApexPlugin> P("APEX", "APEX - Autonomic Performance Environment for eXascale");

// Dynamically loading plugin.
extern "C" RAJA::util::PluginStrategy *getPlugin ()
{
  return new ApexPlugin;
}