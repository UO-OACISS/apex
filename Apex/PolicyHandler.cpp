#include "PolicyHandler.hpp"
#include "ThreadInstance.hpp"

using namespace std;

namespace apex {

PolicyHandler::PolicyHandler (void) : Handler() { 
  _init(); 
}

PolicyHandler::PolicyHandler (unsigned int period) : Handler(period) { 
  _init(); 
}

void PolicyHandler::_handler(void) {
  return;
}

void PolicyHandler::_init(void) {
  _timer.async_wait(boost::bind(&PolicyHandler::_handler, this));
  run();
  return;
}

void PolicyHandler::onEvent(EventData* eventData) {
  unsigned int tid = ThreadInstance::getID();
  if (!_terminate) {
    switch(eventData->eventType) {
      case START_EVENT: 
      {
        break;
      }
      case STOP_EVENT: 
      {
        break;
      }
      case NEW_THREAD: 
      {
        break;
      }
      case SHUTDOWN: 
      {
        _terminate = true;   
        break;
      }
    }
  }
  return;
}

}
