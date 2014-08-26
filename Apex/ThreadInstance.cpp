#include "ThreadInstance.hpp"
#include <iostream>

// TAU related
#define PROFILING_ON 
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>

using namespace std;

namespace apex {

// Global static pointer used to ensure a single instance of the class.
boost::thread_specific_ptr<ThreadInstance> ThreadInstance::_instance;
// Global static count of threads in system
boost::atomic_int ThreadInstance::_numThreads(0);
// Global static map of HPX thread names to TAU thread IDs
map<string, int> ThreadInstance::_nameMap;
// Global static mutex to control access to the map
boost::mutex ThreadInstance::_nameMapMutex;
// Global static map of TAU thread IDs to HPX workers
map<int, bool> ThreadInstance::_workerMap;
// Global static mutex to control access to the map
boost::mutex ThreadInstance::_workerMapMutex;

ThreadInstance* ThreadInstance::Instance(void) {
  ThreadInstance* me = _instance.get();
  if( ! me ) {
    // first time called by this thread
    // construct test element to be used in all subsequent calls from this thread
    _instance.reset( new ThreadInstance());
    me = _instance.get();
    //me->_ID = TAU_PROFILE_GET_THREAD();
    me->_ID = _numThreads++;
  }
  return me;
}

int ThreadInstance::getID(void) {
  return Instance()->_ID;
}

void ThreadInstance::setWorker(bool isWorker) {
  Instance()->_isWorker = isWorker;
  _workerMapMutex.lock();
  _workerMap[Instance()->getID()] = isWorker;
  _workerMapMutex.unlock();
}

string ThreadInstance::getName(void) {
  return *(Instance()->_topLevelTimerName);
}

void ThreadInstance::setName(string name) {
  if (Instance()->_topLevelTimerName == NULL)
  {
    Instance()->_topLevelTimerName = new string(name);
    _nameMapMutex.lock();
    _nameMap[name] = Instance()->getID();
    _nameMapMutex.unlock();
    if (name.find("worker-thread") != name.npos) {
      Instance()->setWorker(true);
    }
  }
}

int ThreadInstance::mapNameToID(string name) {
  //cout << "Looking for " << name << endl;
  int tmp = -1;
  _nameMapMutex.lock();
  if (_nameMap.find(name) != _nameMap.end()) {
    tmp = _nameMap[name];
  }
  _nameMapMutex.unlock();
  return tmp;
}

bool ThreadInstance::mapIDToWorker(int id) {
  //cout << "Looking for " << name << endl;
  bool worker = false;
  _workerMapMutex.lock();
  if (_workerMap.find(id) != _workerMap.end()) {
    worker = _workerMap[id];
  }
  _workerMapMutex.unlock();
  return worker;
}

}
