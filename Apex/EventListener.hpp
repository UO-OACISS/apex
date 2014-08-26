#ifndef APEX_EVENTLISTENER_H
#define APEX_EVENTLISTENER_H

#include <string>

using namespace std;

namespace apex {

/* Typedef for enumerating the different event types */

typedef enum _EventType {
  STARTUP, 
  SHUTDOWN, 
  NEW_NODE, 
  NEW_THREAD, 
  START_EVENT, 
  STOP_EVENT, 
  SAMPLE_VALUE
} EventType;

/* Class for holding data relevant to generic event */

class EventData {
public:
  EventType eventType;
  int threadID;
  EventData() : threadID(0) {};
  ~EventData() {};
};

/* Classes for holding data relevant to specific events */

class TimerEventData : public EventData {
public:
  string * timerName;
  TimerEventData(EventType eventType, int threadID, string timerName);
  ~TimerEventData();
};

class NodeEventData : public EventData {
public:
  int nodeID;
  NodeEventData(int nodeID, int threadID);
  ~NodeEventData() {};
};

class SampleValueEventData : public EventData {
public:
  string * counterName;
  double counterValue;
  SampleValueEventData(int threadID, string counterName, double counterValue);
  ~SampleValueEventData();
};

class StartupEventData : public EventData {
public:
  int argc;
  char** argv;
  StartupEventData(int argc, char** argv);
  ~StartupEventData() {};
};

class ShutdownEventData : public EventData {
public:
  int nodeID;
  ShutdownEventData(int nodeID, int threadID);
  ~ShutdownEventData() {};
};

class NewThreadEventData : public EventData {
public:
  string* threadName;
  NewThreadEventData(string threadName);
  ~NewThreadEventData();
};

/* Abstract class for creating an Event Listener class */

class EventListener
{
public:
  // virtual destructor
  virtual ~EventListener() {};
  // all methods in the interface that a handler has to override
  virtual void onEvent(EventData * eventData) = 0;
};

}

#endif // APEX_EVENTLISTENER_H
