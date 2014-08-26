#include "EventListener.hpp"
#include "ThreadInstance.hpp"

/* At some point, make this multithreaded using the multiproducer/singlecomsumer example
 * at http://www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html
 */

using namespace std;

namespace apex {

TimerEventData::TimerEventData(EventType eventType, int threadID, string timerName) {
  this->eventType = eventType;
  this->threadID = threadID;
  this->timerName = new string(timerName);
}

TimerEventData::~TimerEventData() {
  delete(timerName);
}

NodeEventData::NodeEventData(int nodeID, int threadID) {
  this->eventType = NEW_NODE;
  this->nodeID = nodeID;
  this->threadID = threadID;
}

SampleValueEventData::SampleValueEventData(int threadID, string counterName, double counterValue) {
  this->eventType = SAMPLE_VALUE;
  this->threadID = threadID;
  this->counterName = new string(counterName);
  this->counterValue = counterValue;
}

SampleValueEventData::~SampleValueEventData() {
  delete(counterName);
}

StartupEventData::StartupEventData(int argc, char** argv) {
  this->threadID = ThreadInstance::getID();
  this->eventType = STARTUP;
  this->argc = argc;
  this->argv = argv;
}

ShutdownEventData::ShutdownEventData(int nodeID, int threadID) {
  this->eventType = SHUTDOWN;
  this->nodeID = nodeID;
  this->threadID = threadID;
}

NewThreadEventData::NewThreadEventData(string threadName) {
  this->threadID = ThreadInstance::getID();
  this->eventType = NEW_THREAD;
  this->threadName = new string(threadName);
}

NewThreadEventData::~NewThreadEventData() {
  delete(threadName);
}

}
