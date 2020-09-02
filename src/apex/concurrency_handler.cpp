/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "apex.hpp"
#include "apex_api.hpp"
#include "apex_policies.hpp"
#include "concurrency_handler.hpp"
#include "thread_instance.hpp"
#include <iostream>
#include <map>
#include <iterator>
#include <iostream>
#include <string>
#include <fstream>
#include <atomic>
#include <utility>
#include <memory>
#include <vector>
#include "utils.hpp"

#include "tau_listener.hpp"

#define MAX_FUNCTIONS_IN_CHART 5

using namespace std;

std::vector<std::mutex*> _per_thread_mutex;

namespace apex {

concurrency_handler::concurrency_handler (void) : handler(), _stack_count(0) {
  _init();
}

concurrency_handler::concurrency_handler (int option) :
    handler(), _stack_count(0), _option(option) {
  _init();
}

concurrency_handler::concurrency_handler (unsigned int period) :
    handler(period), _stack_count(0) {
  _init();
}

concurrency_handler::concurrency_handler (unsigned int period, int option) :
    handler(period), _stack_count(0), _option(option) {
  _init();
}

concurrency_handler::~concurrency_handler () {
    for (auto tmp : _event_stack) {
        delete(tmp);
    }
    for (auto tmp : _per_thread_mutex) {
        delete(tmp);
    }
    for (auto tmp : _states) {
        delete(tmp);
    }
}

inline void concurrency_handler::insert_function(task_identifier& func) {
    std::lock_guard<std::mutex> l(_function_mutex);
    if (_functions.find(func) == _functions.end()) {
        _functions.insert(func);
    }
}

bool concurrency_handler::_handler(void) {
  if (!_handler_initialized) {
      initialize_worker_thread_for_tau();
      _handler_initialized = true;
  }
  if (_terminate) return true;
  apex* inst = apex::instance();
  if (inst == nullptr) return false; // running after finalization!
  if (apex_options::use_tau()) {
    tau_listener::Tau_start_wrapper("concurrency_handler::_handler");
  }
  map<task_identifier, unsigned int> *counts = new(map<task_identifier,
    unsigned int>);
  stack<task_identifier>* tmp;
//  std::mutex* mut;
  for (unsigned int i = 0 ; i < _stack_count ; i++) {
    if (_option > 1 && !thread_instance::map_id_to_worker(i)) {
      continue;
    }
    if (inst != nullptr && inst->get_state(i) == APEX_THROTTLED) { continue; }
    tmp = get_event_stack(i);
    task_identifier func;
    if (tmp != nullptr && tmp->size() > 0) {
      {
        std::lock_guard<std::mutex> l(*_per_thread_mutex[i]);
        if (tmp->size() > 0) {
          func = tmp->top();
        } else {
          continue;
        }
      }
      insert_function(func);
      if (counts->find(func) == counts->end()) {
        (*counts)[func] = 1;
      } else {
        (*counts)[func] = (*counts)[func] + 1;
      }
    }
  }
  int power = current_power_high();
  {
    std::lock_guard<std::mutex> l(_vector_mutex);
    _states.push_back(counts);
    _thread_cap_samples.push_back(get_thread_cap());
    // TODO: FIXME multiple tuning sessions
    //for(auto param : get_tunable_params()) {
    //  _tunable_param_samples[param.first].push_back(*param.second);
    //}
    _power_samples.push_back(power);
  }
  if (apex_options::use_tau()) {
    tau_listener::Tau_stop_wrapper("concurrency_handler::_handler");
  }
  if (apex_options::use_jupyter_support()) {
    // update the timeout, if the user changed it.
    set_timeout(apex_options::concurrency_period());
  }
  return true;
}

void concurrency_handler::_init(void) {
  // initialize the vector with ncores elements. For most applications, this
  // should be good enough to avoid data races during initialization.
  add_thread(hardware_concurrency());
  run();
  return;
}

bool concurrency_handler::common_start(task_identifier *id) {
  if (!_terminate) {
    int i = thread_instance::get_id();
    stack<task_identifier>* my_stack = get_event_stack(i);
    std::lock_guard<std::mutex> l(*_per_thread_mutex[i]);
    my_stack->push(*id);
    return true;
  } else {
    return false;
  }
}

bool concurrency_handler::on_start(std::shared_ptr<task_wrapper> &tt_ptr) {
    return common_start(tt_ptr->get_task_id());
}

bool concurrency_handler::on_resume(std::shared_ptr<task_wrapper> &tt_ptr) {
    return common_start(tt_ptr->get_task_id());
}

void concurrency_handler::common_stop(std::shared_ptr<profiler> &p) {
  if (!_terminate) {
    int i = thread_instance::get_id();
    stack<task_identifier>* my_stack = get_event_stack(i);
    std::lock_guard<std::mutex> l(*_per_thread_mutex[i]);
    if (!my_stack->empty()) {
      my_stack->pop();
    }
  }
  APEX_UNUSED(p);
}

void concurrency_handler::on_stop(std::shared_ptr<profiler> &p) {
    common_stop(p);
}

void concurrency_handler::on_yield(std::shared_ptr<profiler> &p) {
    common_stop(p);
}

void concurrency_handler::on_new_thread(new_thread_event_data &data) {
  if (!_terminate) {
        add_thread(data.thread_id);
  }
}

void concurrency_handler::on_exit_thread(event_data &data) {
  APEX_UNUSED(data);
  _terminate = true; // because there are crashes
}

void concurrency_handler::on_dump(dump_event_data &data) {
    output_samples(data.node_id);
    // do something with the data.reset flag
    if (data.reset) {
        reset_samples();
    }
}

void concurrency_handler::on_reset(task_identifier * id) {
    if (id == nullptr) {
        reset_samples();
    }
}

void concurrency_handler::on_shutdown(shutdown_event_data &data) {
    APEX_UNUSED(data);
    _terminate = true; // because there are crashes
    cancel();
}

inline stack<task_identifier>* concurrency_handler::get_event_stack(
    unsigned int tid) {
  // it's possible we could get a "start" event without a "new thread" event.
  stack<task_identifier>* tmp;
  if (tid >= _stack_count) {
    add_thread(tid);
  }
  std::lock_guard<std::mutex> l(_vector_mutex);
  tmp = this->_event_stack[tid];
  return tmp;
}

inline void concurrency_handler::add_thread(unsigned int tid) {
  std::lock_guard<std::mutex> l(_vector_mutex);
  while(_event_stack.size() <= tid) {
    _event_stack.push_back(new stack<task_identifier>);
    _per_thread_mutex.push_back(new std::mutex());
  }
  _stack_count = _event_stack.size();
}

bool sort_functions(pair<task_identifier,int> first,
    pair<task_identifier,int> second) {
  if (first.second > second.second)
    return true;
  return false;
}

void concurrency_handler::reset_samples(void) {
  //cout << "HANDLER: resetting samples " << endl;
  std::lock_guard<std::mutex> l(_vector_mutex);
  _states.clear();
  _power_samples.clear();
  _thread_cap_samples.clear();
}

bool path_has_suffix(const std::string &str)
{
    return str.back() == filesystem_separator();
}

void concurrency_handler::output_samples(int node_id) {
  //cout << "HANDLER: writing samples " << endl;
  cout << _states.size() << " samples seen:" << endl;
  ofstream myfile;
  stringstream datname;
  datname << apex_options::output_file_path();
  if (!path_has_suffix(apex_options::output_file_path())) {
      datname << filesystem_separator();
  }
  datname << "concurrency." << node_id << ".dat";
  myfile.open(datname.str().c_str());
  _function_mutex.lock();
  // limit ourselves to N functions.
  map<task_identifier, int> func_count;
  // initialize the map
  for (set<task_identifier>::iterator it=_functions.begin();
    it!=_functions.end(); ++it) {
    func_count[*it] = 0;
  }
  // count all function instances
  for (unsigned int i = 0 ; i < _states.size() ; i++) {
    for (set<task_identifier>::iterator it=_functions.begin();
        it!=_functions.end(); ++it) {
      if (_states[i]->find(*it) == _states[i]->end()) {
        continue;
      } else {
        func_count[*it] = func_count[*it] + (*(_states[i]))[*it];
      }
    }
  }
  // sort the map
  vector<pair<task_identifier,int> > my_vec(func_count.begin(),
    func_count.end());
  sort(my_vec.begin(),my_vec.end(),&sort_functions);
  set<task_identifier> top_x;
  for (vector<pair<task_identifier, int> >::iterator it=my_vec.begin();
    it!=my_vec.end(); ++it) {
    //if (top_x.size() < 15 && (*it).first != "APEX THREAD MAIN")
    if (top_x.size() < MAX_FUNCTIONS_IN_CHART)
      top_x.insert((*it).first);
  }

  // output the header
  myfile << "\"period\"\t\"thread cap\"\t\"power\"\t";
  // output tunable parameter names
  for(auto param : _tunable_param_samples) {
    myfile << "\"" << param.first << "\"\t";
  }
  for (set<task_identifier>::iterator it=_functions.begin(); it!=_functions.end(); ++it) {
    if (top_x.find(*it) != top_x.end()) {
      task_identifier tmp_id(*it);
      string tmp = tmp_id.get_name();
      myfile << "\"" << tmp << "\"\t";
    }
  }
  myfile << "\"other\"" << endl;

  size_t max_Y = 0;
  double max_Power = 0.0;
  size_t max_X = _states.size();
  int num_params = _tunable_param_samples.size();
  for (size_t i = 0 ; i < max_X ; i++) {
    myfile << i << "\t";
    myfile << _thread_cap_samples[i] << "\t";
    myfile << _power_samples[i] << "\t";
    for(auto param : _tunable_param_samples) {
      myfile << param.second[i] << "\t";
      if(param.second[i] > max_Power) max_Power = param.second[i];
    }
    unsigned int tmp_max = 0;
    int other = 0;
    for (set<task_identifier>::iterator it=_functions.begin();
        it!=_functions.end(); ++it) {
      // this is the idle event.
      //if (*it == "APEX THREAD MAIN")
        //continue;
      int value = 0;
      // did we see this timer during this sample?
      if (_states[i]->find(*it) != _states[i]->end()) {
        value = (*(_states[i]))[*it];
      }
      // is this timer in the top X?
      if (top_x.find(*it) == top_x.end()) {
        other = other + value;
      } else {
        myfile << (*(_states[i]))[*it] << "\t";
        tmp_max += (*(_states[i]))[*it];
      }
    }
    myfile << other << "\t" << endl;
    tmp_max += other;
    if (tmp_max > max_Y) max_Y = tmp_max;
    if ((size_t)(_thread_cap_samples[i]) > max_Y) max_Y = _thread_cap_samples[i];
    if (_power_samples[i] > max_Power) max_Power = _power_samples[i];
  }
  _function_mutex.unlock();
  myfile.close();

  if (max_Power == 0.0) max_Power = 100;
  stringstream plotname;
  plotname << apex_options::output_file_path();
  if (!path_has_suffix(apex_options::output_file_path())) {
      plotname << filesystem_separator();
  }
  plotname << "concurrency." << node_id << ".gnuplot";
  myfile.open(plotname.str().c_str());
  myfile << "set palette maxcolors ";
  myfile << (_functions.size() > 15? 16 : _functions.size()+1) << endl;
  myfile << "set palette defined ( 0 '#E41A1C'";
  myfile << ", 1 '#377EB8'";
  if (_functions.size() > 1)
      myfile << ", 2 '#4DAF4A'";
  if (_functions.size() > 2)
      myfile << ", 3 '#984EA3'";
  if (_functions.size() > 3)
      myfile << ", 4 '#FF7F00'";
  if (_functions.size() > 4)
      myfile << ", 5 '#FFFF33'";
  if (_functions.size() > 5)
      myfile << ", 6 '#A65628'";
  if (_functions.size() > 6)
      myfile << ", 7 '#F781BF'";
  if (_functions.size() > 7)
      myfile << ", 8 '#66C2A5'";
  if (_functions.size() > 8)
      myfile << ", 9 '#FC8D62'";
  if (_functions.size() > 9)
      myfile << ", 10 '#8DA0CB'";
  if (_functions.size() > 10)
      myfile << ", 11 '#E78AC3'";
  if (_functions.size() > 11)
      myfile << ", 12 '#A6D854'";
  if (_functions.size() > 12)
      myfile << ", 13 '#FFD92F'";
  if (_functions.size() > 13)
      myfile << ", 14 '#E5C494'";
  if (_functions.size() > 14)
      myfile << ", 15 '#B3B3B3'";
  myfile << " )" << endl;
  myfile << "set terminal png size 1600,900 font Helvetica 16" << endl;
  myfile << "set output '";
  myfile << apex_options::output_file_path();
  if (!path_has_suffix(apex_options::output_file_path())) {
      myfile << filesystem_separator();
  }
  myfile << "concurrency.png'" << endl;
  myfile << "everyNth(col) = (int(column(col))%" <<
    (int)(max_X/10) << "==0)?stringcolumn(1):\"\"" << endl;
  myfile << "set key outside bottom center invert box" << endl;
  myfile << "set xtics auto nomirror" << endl;
  myfile << "set ytics 4 nomirror" << endl;
  myfile << "set y2tics auto nomirror" << endl;
  myfile << "set xrange[0:" << max_X << "]" << endl;
  myfile << "set yrange[0:" << max_Y << "]" << endl;
  myfile << "set y2range[0:" << max_Power << "]" << endl;
  myfile << "set xlabel \"Time\"" << endl;
  myfile << "set ylabel \"Concurrency\"" << endl;
  myfile << "set y2label \"Power\"" << endl;
  myfile << "# Select histogram data" << endl;
  myfile << "set style data histogram" << endl;
  myfile <<
  "# Give the bars a plain fill pattern, and draw a solid line around them."
    << endl;
  myfile << "set style fill solid border" << endl;
  myfile << "set style histogram rowstacked" << endl;
  myfile << "set boxwidth 1.0 relative" << endl;
  myfile << "#set palette rgb 33,13,10" << endl;
  myfile << "unset colorbox" << endl;
  myfile << "set key noenhanced" << endl; // this allows underscores in names
  myfile << "plot for [COL=" << (4+num_params) << ":" << top_x.size()+num_params+4;
  myfile << "] '" << datname.str().c_str();
  myfile << "' using COL:xticlabel(everyNth(1)) palette frac (COL-" <<
    (3+num_params) << ")/" << top_x.size()+1;
  myfile << ". title columnheader axes x1y1, '"  << datname.str().c_str();
  myfile << "' using 2 with lines linecolor rgb \"red\" axes x1y1 title columnheader, '"
    << datname.str().c_str();
  myfile << "' using 3 with lines linecolor rgb \"black\" axes x1y2 title columnheader";
  if (num_params > 0) {
    myfile << ",";
  }
  for(int p = 0; p < num_params; ++p) {
    myfile << "'" << datname.str().c_str() << "' using " << (4+p) <<
        " with linespoints axes x1y2 title columnheader,";
  }
  myfile << endl;
  myfile.close();
}

}
