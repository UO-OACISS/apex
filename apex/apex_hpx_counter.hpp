#ifndef APEX_HPX_COUNTER_HPP
#define APEX_HPX_COUNTER_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/performance_counters/base_performance_counter.hpp>

namespace apex {
  
  template<typename T>
  class apex_hpx_counter : public hpx::performance_counters::base_performance_counter<apex_hpx_counter<T>> {
    public:
      apex_hpx_counter(std::function<T(void)> f) : value_function(f), invocation_count(0) {};
      hpx::performance_counters::counter_value get_counter_value(bool reset);
      bool start();
      bool stop();
      void finalize();
    private:
      std::function<T(void)> value_function;
      int64_t invocation_count;
  };

}

#endif
