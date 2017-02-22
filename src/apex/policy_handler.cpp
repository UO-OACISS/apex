//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef APEX_HAVE_HPX
#include <hpx/config.hpp>
#include <hpx/include/runtime.hpp>
#endif

#include "apex_api.hpp"
#include "apex.hpp"
#include "policy_handler.hpp"
#include <iostream>
#include <atomic>
#include <unistd.h>
#include "tau_listener.hpp"

using namespace std;

namespace apex {

	std::atomic<int> next_id(0);

#ifdef APEX_HAVE_HPX
	policy_handler::policy_handler (void) : handler() { }
#else
	policy_handler::policy_handler (void) : handler() { }
#endif

	/*
	   template <typename Rep, typename Period>
	   policy_handler::policy_handler (duration<Rep, Period> const& period) : handler(period)
	   {
	   _init();
	   }
	   */

#ifdef APEX_HAVE_HPX
	policy_handler::policy_handler (uint64_t period_microseconds) : handler(period_microseconds), hpx_timer(boost::bind(&policy_handler::_handler, this), _period, "apex_internal_policy_handler") 
	{
		_init();
	}
#else
	policy_handler::policy_handler (uint64_t period_microseconds) : handler(period_microseconds) 
	{
		_init();
	}
#endif

	bool policy_handler::_handler(void) {
		if (!_handler_initialized) {
			initialize_worker_thread_for_tau();
			_handler_initialized = true;
		}
		this->_reset();
		if (_terminate) return true;
		if (apex_options::use_tau()) {
			Tau_start("policy_handler::_handler");
		}
		periodic_event_data data;
		this->on_periodic(data);
		if (apex_options::use_tau()) {
			Tau_stop("policy_handler::_handler");
		}
		return true;
	}

	void policy_handler::_init(void) {
#ifdef APEX_HAVE_HPX
		hpx_timer.start();
#else
		run();
#endif
		return;
	}

	inline void policy_handler::_reset(void) {
#ifdef APEX_HAVE_HPX
		if (_terminate) {
			hpx_timer.stop();
		}
#endif
	}


	int policy_handler::register_policy(const apex_event_type & when,
			std::function<int(apex_context const&)> f) {
        // save the old policy setting
        bool old_policy_setting = apex_options::use_policy();
        // prevent policies from iterating - kind of like a lock, but faster.
        apex_options::use_policy(false);
        // Sleep just a tiny bit - that allows other threads that might be
        // currently processing policies to clear out. This is actually 
        // more efficient than a lock transaction on every policy execution.
        usleep(apex_options::policy_drain_timeout()); // sleep 1ms
		int id = next_id++;
		std::shared_ptr<policy_instance> instance(
				std::make_shared<policy_instance>(id, f));
		switch(when) {
			case APEX_STARTUP: {
						   //write_lock_type l(startup_mutex);
						   startup_policies.push_back(instance);
						   break;
					   }
			case APEX_SHUTDOWN: {
							//write_lock_type l(shutdown_mutex);
							shutdown_policies.push_back(instance);
							break;
						}
			case APEX_NEW_NODE: {
							//write_lock_type l(new_node_mutex);
							new_node_policies.push_back(instance);
							break;
						}
			case APEX_NEW_THREAD: {
							  //write_lock_type l(new_thread_mutex);
							  new_thread_policies.push_back(instance);
							  break;
						  }
			case APEX_EXIT_THREAD: {
							   //write_lock_type l(exit_thread_mutex);
							   exit_thread_policies.push_back(instance);
							   break;
						   }
			case APEX_START_EVENT: {
							   //write_lock_type l(start_event_mutex);
							   start_event_policies.push_back(instance);
							   break;
						   }
			case APEX_RESUME_EVENT: {
							//write_lock_type l(resume_event_mutex);
							resume_event_policies.push_back(instance);
							break;
						}
			case APEX_STOP_EVENT: {
							  //write_lock_type l(stop_event_mutex);
							  stop_event_policies.push_back(instance);
							  break;
						  }
			case APEX_YIELD_EVENT: {
							   //write_lock_type l(yield_event_mutex);
							   yield_event_policies.push_back(instance);
							   break;
						   }
			case APEX_SAMPLE_VALUE: {
							//write_lock_type l(sample_value_mutex);
							sample_value_policies.push_back(instance);
							break;
						}
			case APEX_PERIODIC: {
							//write_lock_type l(periodic_mutex);
							periodic_policies.push_back(instance);
							break;
						}
						//case APEX_CUSTOM_EVENT_1:
			default: {
					 //write_lock_type l(custom_event_mutex);
					 custom_event_policies[when].push_back(instance);
					 break;
				 }
		}
        apex_options::use_policy(old_policy_setting);
		return id;

	}

	int policy_handler::deregister_policy(apex_policy_handle * handle) {
        // save the old policy setting
        bool old_policy_setting = apex_options::use_policy();
        // prevent policies from iterating - kind of like a lock, but faster.
        apex_options::use_policy(false);
        // Sleep just a tiny bit - that allows other threads that might be
        // currently processing policies to clear out. This is actually 
        // more efficient than a lock transaction on every policy execution.
        usleep(apex_options::policy_drain_timeout()); // sleep 1ms
		switch(handle->event_type) {
			case APEX_STARTUP: {
						   //write_lock_type l(startup_mutex);
						   std::list<std::shared_ptr<policy_instance> >::iterator it;
						   for(it = startup_policies.begin() ; it != startup_policies.end() ; it++) {
							   std::shared_ptr<policy_instance> policy = *it;
							   if (policy->id == handle->id) {
								   startup_policies.erase(it);
								   break;
							   }
						   }
						   break;
					   }
			case APEX_SHUTDOWN: {
							//write_lock_type l(shutdown_mutex);
							std::list<std::shared_ptr<policy_instance> >::iterator it;
							for(it = shutdown_policies.begin() ; it != shutdown_policies.end() ; it++) {
								std::shared_ptr<policy_instance> policy = *it;
								if (policy->id == handle->id) {
									shutdown_policies.erase(it);
									break;
								}
							}
							break;
						}
			case APEX_NEW_NODE: {
							//write_lock_type l(new_node_mutex);
							std::list<std::shared_ptr<policy_instance> >::iterator it;
							for(it = new_node_policies.begin() ; it != new_node_policies.end() ; it++) {
								std::shared_ptr<policy_instance> policy = *it;
								if (policy->id == handle->id) {
									new_node_policies.erase(it);
									break;
								}
							}
							break;
						}
			case APEX_NEW_THREAD: {
							  //write_lock_type l(new_thread_mutex);
							  std::list<std::shared_ptr<policy_instance> >::iterator it;
							  for(it = new_thread_policies.begin() ; it != new_thread_policies.end() ; it++) {
								  std::shared_ptr<policy_instance> policy = *it;
								  if (policy->id == handle->id) {
									  new_thread_policies.erase(it);
									  break;
								  }
							  }
							  break;
						  }
			case APEX_EXIT_THREAD: {
							   //write_lock_type l(exit_thread_mutex);
							   std::list<std::shared_ptr<policy_instance> >::iterator it;
							   for(it = exit_thread_policies.begin() ; it != exit_thread_policies.end() ; it++) {
								   std::shared_ptr<policy_instance> policy = *it;
								   if (policy->id == handle->id) {
									   exit_thread_policies.erase(it);
									   break;
								   }
							   }
							   break;
						   }
			case APEX_START_EVENT: {
							   //write_lock_type l(start_event_mutex);
							   std::list<std::shared_ptr<policy_instance> >::iterator it;
							   for(it = start_event_policies.begin() ; it != start_event_policies.end() ; it++) {
								   std::shared_ptr<policy_instance> policy = *it;
								   if (policy->id == handle->id) {
									   start_event_policies.erase(it);
									   break;
								   }
							   }
							   break;
						   }
			case APEX_RESUME_EVENT: {
							//write_lock_type l(resume_event_mutex);
							std::list<std::shared_ptr<policy_instance> >::iterator it;
							for(it = resume_event_policies.begin() ; it != resume_event_policies.end() ; it++) {
								std::shared_ptr<policy_instance> policy = *it;
								if (policy->id == handle->id) {
									resume_event_policies.erase(it);
									break;
								}
							}
							break;
						}
			case APEX_STOP_EVENT: {
							  //write_lock_type l(stop_event_mutex);
							  std::list<std::shared_ptr<policy_instance> >::iterator it;
							  for(it = stop_event_policies.begin() ; it != stop_event_policies.end() ; it++) {
								  std::shared_ptr<policy_instance> policy = *it;
								  if (policy->id == handle->id) {
									  stop_event_policies.erase(it);
									  break;
								  }
							  }
							  break;
						  }
			case APEX_YIELD_EVENT: {
							   //write_lock_type l(yield_event_mutex);
							   std::list<std::shared_ptr<policy_instance> >::iterator it;
							   for(it = yield_event_policies.begin() ; it != yield_event_policies.end() ; it++) {
								   std::shared_ptr<policy_instance> policy = *it;
								   if (policy->id == handle->id) {
									   yield_event_policies.erase(it);
									   break;
								   }
							   }
							   break;
						   }
			case APEX_SAMPLE_VALUE: {
							//write_lock_type l(sample_value_mutex);
							std::list<std::shared_ptr<policy_instance> >::iterator it;
							for(it = sample_value_policies.begin() ; it != sample_value_policies.end() ; it++) {
								std::shared_ptr<policy_instance> policy = *it;
								if (policy->id == handle->id) {
									sample_value_policies.erase(it);
									break;
								}
							}
							break;
						}
			case APEX_PERIODIC: {
							//write_lock_type l(periodic_mutex);
							std::list<std::shared_ptr<policy_instance> >::iterator it;
							for(it = periodic_policies.begin() ; it != periodic_policies.end() ; it++) {
								std::shared_ptr<policy_instance> policy = *it;
								if (policy->id == handle->id) {
									periodic_policies.erase(it);
									break;
								}
							}
							break;
						}
						//case APEX_CUSTOM_EVENT_1:
			default: {
					 //write_lock_type l(custom_event_mutex);
					 std::list<std::shared_ptr<policy_instance> >::iterator it;
					 for(it = custom_event_policies[handle->event_type].begin() ; it != custom_event_policies[handle->event_type].end() ; it++) {
						 std::shared_ptr<policy_instance> policy = *it;
						 if (policy->id == handle->id) {
							 custom_event_policies[handle->event_type].erase(it);
							 break;
						 }
					 }
					 break;
				 }
		}
        apex_options::use_policy(old_policy_setting);
		return APEX_NOERROR;
	}

	inline void policy_handler::call_policies(
			const std::list<std::shared_ptr<policy_instance> > & policies,
			void *data, const apex_event_type& event_type) {
        /* we are using this flag to prevent race conditions on the
         * list of policies. If the list changes, we can't iterate
         * over it. */
        if (!apex_options::use_policy()) { return; }
		for(const std::shared_ptr<policy_instance>& policy : policies) {
			apex_context my_context;
			my_context.event_type = event_type;
			my_context.policy_handle = NULL;
			my_context.data = data;
			// last chance to interrupt policy execution at shutdown
			if (_terminate) return;
			const bool result = policy->func(my_context);
			if(result != APEX_NOERROR) {
				printf("Warning: registered policy function failed!\n");
			}
            if (!apex_options::use_policy()) { return; }
		}
	}

	void policy_handler::on_startup(startup_event_data &data) {
		call_policies(startup_policies, (void *)&data, data.event_type_);
	}

	void policy_handler::on_shutdown(shutdown_event_data &data) {
		if (_terminate) return;
		// prevent periodic policies from executing while we are shutting down.
		_terminate = true;
#ifndef APEX_HAVE_HPX
		cancel();
#endif
        if (!apex_options::use_policy()) { return; }
		for(const std::shared_ptr<policy_instance>& policy : shutdown_policies) {
			apex_context my_context;
			my_context.event_type = APEX_SHUTDOWN;
			my_context.policy_handle = NULL;
			my_context.data = NULL;
			const bool result = policy->func(my_context);
			if(result != APEX_NOERROR) {
				printf("Warning: registered policy function failed!\n");
			}
            if (!apex_options::use_policy()) { return; }
		}
	}

	void policy_handler::on_new_node(node_event_data &data) {
		call_policies(new_node_policies, (void *)&data, APEX_NEW_NODE);
	}

	void policy_handler::on_new_thread(new_thread_event_data &data) {
		call_policies(new_thread_policies, (void *)&data, APEX_NEW_THREAD);
	}

	void policy_handler::on_exit_thread(event_data &data) {
		call_policies(exit_thread_policies, (void *)&data, APEX_EXIT_THREAD);
	}

	bool policy_handler::on_start(task_identifier *id) {
		call_policies(start_event_policies, (void *)id, APEX_START_EVENT);
		return true;
	}

	bool policy_handler::on_resume(task_identifier * id) {
		call_policies(resume_event_policies, (void *)id, APEX_RESUME_EVENT);
		return true;
	}

	void policy_handler::on_stop(std::shared_ptr<profiler> &p) {
		call_policies(stop_event_policies, (void *)p->task_id, APEX_STOP_EVENT);
	}

	void policy_handler::on_yield(std::shared_ptr<profiler> &p) {
		call_policies(yield_event_policies, (void *)p->task_id, APEX_YIELD_EVENT);
	}

	void policy_handler::on_sample_value(sample_value_event_data &data) {
		call_policies(sample_value_policies, &data, APEX_SAMPLE_VALUE);
	}

	void policy_handler::on_custom_event(custom_event_data &data) {
        if (!apex_options::use_policy()) { return; }
		for(const std::shared_ptr<policy_instance>& policy : custom_event_policies[data.event_type_]) {
			apex_context my_context;
			my_context.event_type = data.event_type_;
			my_context.policy_handle = NULL;
			my_context.data = data.data;
			// last chance to interrupt policy execution at shutdown
			// HOWEVER, if the event is shutdown, run the policy.
			if (_terminate) return;
			const bool result = policy->func(my_context);
			if(result != APEX_NOERROR) {
				printf("Warning: registered policy function failed!\n");
			}
            if (!apex_options::use_policy()) { return; }
		}
	}

	void policy_handler::on_periodic(periodic_event_data &data) {
		// get the read lock first. Because if we are shutting down,
		// the _terminate flag will be set after the write lock has 
		// been acquired.  Yes, the periodic lock will be held a few 
		// instructions longer, but this prevents race condition
		// SegVs during shutdown.
		call_policies(periodic_policies, (void *)&data, APEX_PERIODIC);
	}

} // end namespace apex

