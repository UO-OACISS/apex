#pragma once

#include "RAJA/util/PluginStrategy.hpp"
#include "apex_api.hpp"
#include <stack>
#include <sstream>

class ApexPlugin : public RAJA::util::PluginStrategy {
  public:
  void init(const RAJA::util::PluginOptions& p) override {
    APEX_UNUSED(p);
    apex::init("APEX: Raja support", 0, 1);
  }
  void finalize() override {
    apex::finalize();
  }

  void preCapture(const RAJA::util::PluginContext& p) override {
    apex::profiler * prof;
    std::stringstream ss;
    if (p.platform == RAJA::Platform::host) {
        ss << "RAJA host capture"; // << p.kID;
    } else {
        ss << "RAJA device capture"; // << p.kID;
    }
    std::string tmp{ss.str()};
    prof = apex::start(tmp);
    get_stack().push(prof);
  }

  void postCapture(const RAJA::util::PluginContext& p) override {
    APEX_UNUSED(p);
    apex::profiler * prof = get_stack().top();
    apex::stop(prof);
    get_stack().pop();
  }

  void preLaunch(const RAJA::util::PluginContext& p) override {
    apex::profiler * prof;
    if (p.platform == RAJA::Platform::host) {
        prof = apex::start("RAJA host kernel");
    } else {
        prof = apex::start("RAJA device kernel");
    }
    get_stack().push(prof);
  }

  void postLaunch(const RAJA::util::PluginContext& p) override {
    APEX_UNUSED(p);
    apex::profiler * prof = get_stack().top();
    apex::stop(prof);
    get_stack().pop();
  }

  private:
    static std::stack<apex::profiler*>& get_stack() {
        static APEX_NATIVE_TLS std::stack<apex::profiler*> timer_stack;
        return timer_stack;
    }
};
