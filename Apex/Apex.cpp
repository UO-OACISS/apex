#include "Apex.hpp"
#include "ApexConfig.h"
#ifdef APEX_HAVE_RCR
#include "energyStat.h"
#endif
#include <iostream>
#include <stdlib.h>
#include <string>
//#include <cxxabi.h>

#include "ConcurrencyHandler.hpp"
#include "PolicyHandler.hpp"
#include "TauListener.hpp"
#include "ThreadInstance.hpp"

#define PROFILING_ON 
//#define TAU_GNU 
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>

#if 0
#define APEX_TRACER {int __nid = TAU_PROFILE_GET_NODE(); \
 int __tid = ThreadInstance::getID(); \
cout << __nid << ":" << __tid << " " << __FUNCTION__ << " ["<< __FILE__ << ":" << __LINE__ << "]" << endl;}
#else
#define APEX_TRACER 
#endif

#if 0
#define APEX_TIMER_TRACER(A, B) {int __nid = TAU_PROFILE_GET_NODE(); \
 int __tid = TAU_PROFILE_GET_THREAD(); \
cout << __nid << ":" << __tid << " " << (A) << " "<< (B) << endl;}
#else
#define APEX_TIMER_TRACER(A, B)
#endif

using namespace std;

namespace apex {

// Global static pointer used to ensure a single instance of the class.
Apex* Apex::m_pInstance = NULL; 

static bool _notifyListeners = true;
static bool _finalized = false;

/** The destructor will request power data from RCRToolkit
 **/
Apex::~Apex() {
  APEX_TRACER
#ifdef APEX_HAVE_RCR
  cout << "Getting energy..." << endl;
  energyDaemonTerm();
#endif
  m_pInstance = NULL; 
}

void Apex::setNodeID(int id) {
  APEX_TRACER
  m_node_id = id;
  stringstream ss;
  ss << "locality#" << m_node_id;
  m_my_locality = new string(ss.str());
  NodeEventData* eventData = new NodeEventData(0, id);
  this->notifyListeners(eventData);
}

int Apex::getNodeID() {
  APEX_TRACER
  return m_node_id;
}

/* 
 * This private method is used to perform whatever initialization
 * needs to happen.
 */
void Apex::_initialize() {
  APEX_TRACER
  this->m_pInstance = this;
#ifdef APEX_HAVE_RCR
    uint64_t waitTime = 1000000000L; // in nanoseconds, for nanosleep
    energyDaemonInit(waitTime);
#endif
  char* option = NULL;
#if APEX_HAVE_TAU
  option = getenv("APEX_TAU");
  if (option != NULL) {
    listeners.push_back(new TauListener());
  }
#endif
  option = getenv("APEX_POLICY");
  if (option != NULL) {
    listeners.push_back(new PolicyHandler());
  }
  option = getenv("APEX_CONCURRENCY");
  if (option != NULL && atoi(option) > 0) {
    char* option2 = getenv("APEX_CONCURRENCY_PERIOD");
    if (option2 != NULL) {
      listeners.push_back(new ConcurrencyHandler(atoi(option2), option));
	} else {
      listeners.push_back(new ConcurrencyHandler(option));
	}
  }
  setNodeID(0);
}

/** This function is called to create an instance of the class.
    Calling the constructor publicly is not allowed. The constructor
    is private and is only called by this Instance function.
*/
Apex* Apex::Instance()
{
  //APEX_TRACER
  // Only allow one instance of class to be generated.
  if (m_pInstance == NULL && !_finalized) {
    m_pInstance = new Apex;
  }
  return m_pInstance;
}

Apex* Apex::Instance(int argc, char**argv)
{
  //APEX_TRACER
  // Only allow one instance of class to be generated.
  if (m_pInstance == NULL && !_finalized) {
    m_pInstance = new Apex(argc, argv);
  }
  return m_pInstance;
}

void Apex::notifyListeners(EventData* eventData)
{
  if (_notifyListeners) {
    for (unsigned int i = 0 ; i < listeners.size() ; i++) {
      listeners[i]->onEvent(eventData);
    }
  }
}

void init() {
  APEX_TRACER
  int argc = 1;
  const char *dummy = "APEX Application";
  char* argv[1];
  argv[0] = const_cast<char*>(dummy);
  Apex* instance = Apex::Instance(); // get/create the Apex static instance
  if (!instance) return; // protect against calls after finalization
  //TAU_PROFILE_INIT(argc, argv);
  StartupEventData* eventData = new StartupEventData(argc, argv);
  instance->notifyListeners(eventData);
  //start("APEX THREAD MAIN");
}

void init(int argc, char** argv) {
  APEX_TRACER
  Apex* instance = Apex::Instance(argc, argv); // get/create the Apex static instance
  if (!instance) return; // protect against calls after finalization
  //TAU_PROFILE_INIT(argc, argv);
  StartupEventData* eventData = new StartupEventData(argc, argv);
  instance->notifyListeners(eventData);
  //start("APEX THREAD MAIN");
}

double version() {
  APEX_TRACER
  return Apex_VERSION_MAJOR + (Apex_VERSION_MINOR/10.0);
}

/*
string* demangle(string timer_name) {
  int     status;
  string* demangled = NULL;
  char *realname = abi::__cxa_demangle(timer_name.c_str(), 0, 0, &status);
  if (status == 0) {
    char* index = strstr(realname, "<");
    if (index != NULL) {
	  *index = 0; // terminate before templates for brevity
	}
    demangled = new string(realname);
	free(realname);
  }
  return demangled;
}
*/

void start(string timer_name) {
  APEX_TIMER_TRACER("start", timer_name)
  Apex* instance = Apex::Instance(); // get the Apex static instance
  if (!instance) return; // protect against calls after finalization
  //TAU_START(timer_name.c_str());
  EventData* eventData = NULL;
  // don't do this now
  /* 
  string* demangled = NULL;
  demangled = demangle(timer_name);
  if (demangled != NULL) {
    eventData = new TimerEventData(START_EVENT, ThreadInstance::getID(), *demangled);
	delete(demangled);
  } else {
    eventData = new TimerEventData(START_EVENT, ThreadInstance::getID(), timer_name);
  }
  */
  eventData = new TimerEventData(START_EVENT, ThreadInstance::getID(), timer_name);
  instance->notifyListeners(eventData);
}

void stop(string timer_name) {
  APEX_TIMER_TRACER("stop", timer_name)
  Apex* instance = Apex::Instance(); // get the Apex static instance
  if (!instance) return; // protect against calls after finalization
  //TAU_STOP(timer_name.c_str());
  EventData* eventData = NULL;
  // don't do this now
  /* 
  string* demangled = demangle(timer_name);
  if (demangled != NULL) {
    eventData = new TimerEventData(STOP_EVENT, ThreadInstance::getID(), *demangled);
	delete(demangled);
  } else {
    eventData = new TimerEventData(STOP_EVENT, ThreadInstance::getID(), timer_name);
  }
  */
  eventData = new TimerEventData(STOP_EVENT, ThreadInstance::getID(), timer_name);
  instance->notifyListeners(eventData);
}

void stop() {
  APEX_TIMER_TRACER("stop", "?")
  Apex* instance = Apex::Instance(); // get the Apex static instance
  if (!instance) return; // protect against calls after finalization
  //TAU_GLOBAL_TIMER_STOP(); // stop the top level timer
  string empty = "";
  EventData* eventData = new TimerEventData(STOP_EVENT, ThreadInstance::getID(), empty);
  instance->notifyListeners(eventData);
}

void sample_value(string name, double value) {
  APEX_TRACER
  Apex* instance = Apex::Instance(); // get the Apex static instance
  if (!instance) return; // protect against calls after finalization
  // parse the counter name
  // either /threadqueue{locality#0/total}/length
  // or     /threadqueue{locality#0/worker-thread#0}/length
  SampleValueEventData* eventData = NULL;
  if (name.find(*(instance->m_my_locality)) != name.npos) {
    if (name.find("worker-thread") != name.npos) {
      string tmpName = string(name.c_str());
      // tokenize by / character
      char* token = strtok((char*)tmpName.c_str(), "/");
      while (strstr(token, "worker-thread")==NULL) {
      	token = strtok(NULL, "/");
      }
      // strip the trailing close bracket
      token = strtok(token, "}");
      int tid = ThreadInstance::mapNameToID(token);
      if (tid != -1) {
        eventData = new SampleValueEventData(tid, name, value);
        //Tau_trigger_context_event_thread((char*)name.c_str(), value, tid);
      } else {
        eventData = new SampleValueEventData(0, name, value);
        //Tau_trigger_context_event_thread((char*)name.c_str(), value, 0);
      }
    } else {
      eventData = new SampleValueEventData(0, name, value);
      //Tau_trigger_context_event_thread((char*)name.c_str(), value, 0);
      //TAU_TRIGGER_CONTEXT_EVENT((char *)(name.c_str()), value);
    }
  } else { 
  // what if it doesn't?
    eventData = new SampleValueEventData(0, name, value);
    //TAU_TRIGGER_CONTEXT_EVENT((char *)(name.c_str()), value);
  }
  instance->notifyListeners(eventData);
}

void set_node_id(int id) {
  APEX_TRACER
  Apex* instance = Apex::Instance();
  if (!instance) return; // protect against calls after finalization
  instance->setNodeID(id);
}

void track_power(void) {
  APEX_TRACER
  TAU_TRACK_POWER();
}

void track_power_here(void) {
  APEX_TRACER
  TAU_TRACK_POWER_HERE();
}

void enable_tracking_power(void) {
  APEX_TRACER
  TAU_ENABLE_TRACKING_POWER();
}

void disable_tracking_power(void) {
  APEX_TRACER
  TAU_DISABLE_TRACKING_POWER();
}

void set_interrupt_interval(int seconds) {
  APEX_TRACER
  TAU_SET_INTERRUPT_INTERVAL(seconds);
}

void finalize() {
  APEX_TRACER
  Apex* instance = Apex::Instance(); // get the Apex static instance
  if (!instance) return; // protect against calls after finalization
  // exit ALL threads
  //Tau_profile_exit_all_threads();
  //TAU_PROFILE_EXIT("APEX exiting");
  if (!_finalized) {
    _finalized = true;
    stringstream ss;
    ss << instance->getNodeID();
    ShutdownEventData* eventData = new ShutdownEventData(instance->getNodeID(), ThreadInstance::getID());
    instance->notifyListeners(eventData);
    _notifyListeners = false;
  }
  instance->~Apex();
}

void register_thread(string name) {
  APEX_TRACER
  Apex* instance = Apex::Instance(); // get the Apex static instance
  if (!instance) return; // protect against calls after finalization
  //TAU_REGISTER_THREAD();
  // int nid, tid;
  // nid = TAU_PROFILE_GET_NODE();
  // tid = TAU_PROFILE_GET_THREAD();
  //cout << "Node " << nid << " registered thread " << tid << endl;
  // we can't start, because there is no way to stop!
  ThreadInstance::setName(name);
  NewThreadEventData* eventData = new NewThreadEventData(name);
  instance->notifyListeners(eventData);
  string::size_type index = name.find("#");
  if (index!=std::string::npos) {
    string shortName = name.substr(0,index);
    cout << "shortening " << name << " to " << shortName << endl;
    start(shortName);
  } else {
    start(name);
  }
}


} // apex namespace

using namespace apex;

extern "C" {

void apex_init(int argc, char** argv) {
  APEX_TRACER
  init(argc, argv);
}

void apex_finalize() {
  APEX_TRACER
  finalize();
}

double apex_version() {
  APEX_TRACER
  return version();
}

void apex_start(const char * timer_name) {
  APEX_TRACER
  start(string(timer_name));
}

void apex_stop(const char * timer_name) {
  APEX_TRACER
  stop(string(timer_name));
}

void apex_sample_value(const char * name, double value) {
  APEX_TRACER
  sample_value(string(name), value);
}

void apex_set_node_id(int id) {
  APEX_TRACER
  set_node_id(id);
}

void apex_register_thread(const char * name) {
  APEX_TRACER
  register_thread(string(name));
}
void apex_track_power(void) {
  track_power();
}

void apex_track_power_here(void) {
  track_power_here();
}

void apex_enable_tracking_power(void) {
  enable_tracking_power();
}

void apex_disable_tracking_power(void) {
  disable_tracking_power();
}

void apex_set_interrupt_interval(int seconds) {
  set_interrupt_interval(seconds);
}


} // extern "C"

