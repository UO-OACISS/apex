// apex main class
#ifndef APEX_HPP
#define APEX_HPP

#include <string>
#include <vector>
#include <stdint.h>
#include "Handler.hpp"
#include "EventListener.hpp"

using namespace std;

namespace apex {

class Apex {
private:
// private constructors cannot be called
  Apex() : m_argc(0), m_argv(NULL), m_node_id(0) {_initialize();}; 
  Apex(int argc, char**argv) : m_argc(argc), m_argv(argv) {_initialize();};
  Apex(Apex const&){};             // copy constructor is private
  Apex& operator=(Apex const& a){ return const_cast<Apex&>(a); };  // assignment operator is private
  static Apex* m_pInstance;
  int m_argc;
  char** m_argv;
  int m_node_id;
  bool m_profiling;
  void _initialize();
  vector<EventListener*> listeners;
public:
  string* m_my_locality;
  static Apex* Instance(); // singleton instance
  static Apex* Instance(int argc, char** argv); // singleton instance
  void setNodeID(int id);
  int getNodeID(void);
  void notifyListeners(EventData* eventData);
  ~Apex();
};

void init(void);
void init(int argc, char** argv);
void finalize(void);
double version(void);
void start(std::string timer_name);
void stop(std::string timer_name);
void stop(void);
void sample_value(std::string name, double value);
void set_node_id(int id);
void register_thread(std::string name);
void track_power(void);
void track_power_here(void);
void enable_tracking_power(void);
void disable_tracking_power(void);
void set_interrupt_interval(int seconds);
}

#endif //APEX_HPP
