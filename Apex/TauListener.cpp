#include "TauListener.hpp"
#include "ThreadInstance.hpp"
#include <iostream>
#include <fstream>

#define PROFILING_ON 
//#define TAU_GNU 
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>

extern "C" int Tau_profile_exit_all_tasks();
extern "C" int Tau_profile_exit_all_threads();

using namespace std;

namespace apex {

TauListener::TauListener (void) : _terminate(false) { 
}

void TauListener::onEvent(EventData * eventData) {
  unsigned int tid = ThreadInstance::getID();
  if (!_terminate) {
    if (eventData->eventType == START_EVENT) {
      TimerEventData *tmp = (TimerEventData*)eventData;
      TAU_START(tmp->timerName->c_str());
    } else if (eventData->eventType == STOP_EVENT) {
      TimerEventData *tmp = (TimerEventData*)eventData;
      if (*(tmp->timerName) == string("")) {
        TAU_GLOBAL_TIMER_STOP(); // stop the top level timer
      } else {
        TAU_STOP(tmp->timerName->c_str());
      }
    } else if (eventData->eventType == NEW_NODE) {
      NodeEventData *tmp = (NodeEventData*)eventData;
      TAU_PROFILE_SET_NODE(tmp->nodeID);
    } else if (eventData->eventType == NEW_THREAD) {
      TAU_REGISTER_THREAD();
      // set the thread id for future listeners to this event
      eventData->threadID = TAU_PROFILE_GET_THREAD();
    } else if (eventData->eventType == SAMPLE_VALUE) {
      SampleValueEventData *tmp = (SampleValueEventData*)eventData;
      Tau_trigger_context_event_thread((char*)tmp->counterName->c_str(), tmp->counterValue, tmp->threadID);
    } else if (eventData->eventType == STARTUP) {
      StartupEventData *tmp = (StartupEventData*)eventData;
      TAU_PROFILE_INIT(tmp->argc, tmp->argv);
    } else if (eventData->eventType == SHUTDOWN) {
      ShutdownEventData *tmp = (ShutdownEventData*)eventData;
      _terminate = true;   
      Tau_profile_exit_all_threads();
      TAU_PROFILE_EXIT("APEX exiting");
    }
  }
  return;
}

}
