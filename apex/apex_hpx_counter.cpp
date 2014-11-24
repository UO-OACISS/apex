#include <hpx/config.hpp>
#include "apex_hpx_counter.hpp"

namespace apex { 
    
    template<typename T>
    hpx::performance_counters::counter_value apex_hpx_counter<T>::get_counter_value(bool reset) {
      const boost::int64_t scaling = 100000;
      hpx::performance_counters::counter_value value;
      value.value_         = boost::int64_t(value_function() * 100000);
      value.time_          = static_cast<boost::int64_t>(hpx::get_system_uptime());
      value.scaling_       = scaling;
      value.scale_inverse_ = true;
      value.status_        = hpx::performance_counters::status_new_data;
      value.count_         = ++invocation_count;
      return value;
    }

    
    template<typename T> bool apex_hpx_counter<T>::start() {return true;}
    template<typename T> bool apex_hpx_counter<T>::stop() {return true;}
    template<typename T> void apex_hpx_counter<T>::finalize() {
      hpx::performance_counters::base_performance_counter<apex_hpx_counter<T>>::finalize();
    }

} //namespace apex
