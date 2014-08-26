#include "ConcurrencyHandler.hpp"
#include "ThreadInstance.hpp"
#include <iostream>
#include <map>
#include <iterator>
#include <iostream>
#include <fstream>
#include <cxxabi.h>

using namespace std;

namespace apex {

ConcurrencyHandler::ConcurrencyHandler (void) : Handler() { 
  _init(); 
}

ConcurrencyHandler::ConcurrencyHandler (char *option) : Handler(), _option(atoi(option)) { 
  _init(); 
}

ConcurrencyHandler::ConcurrencyHandler (unsigned int period) : Handler(period) { 
  _init(); 
}

ConcurrencyHandler::ConcurrencyHandler (unsigned int period, char *option) : Handler(period), _option(atoi(option)) { 
  _init(); 
}

void ConcurrencyHandler::_handler(void) {
  //cout << "HANDLER: " << endl;
  map<string, unsigned int> *counts = new(map<string, unsigned int>);
  stack<string>* tmp;
  boost::mutex* mut;
  for (unsigned int i = 0 ; i < _eventStack.size() ; i++) {
    if (_option > 1 && !ThreadInstance::mapIDToWorker(i)) {
      continue;
    }
    tmp = getEventStack(i);
    if (tmp->size() > 0) {
	  mut = getEventStackMutex(i);
      mut->lock();
      string func = tmp->top();
      mut->unlock();
      _functionMutex.lock();
      _functions.insert(func);
      _functionMutex.unlock();
      if (counts->find(func) == counts->end()) {
        (*counts)[func] = 1;
      } else {
        (*counts)[func] = (*counts)[func] + 1;
      }
    }
  }
  _states.push_back(counts);
  this->reset();
  return;
}

void ConcurrencyHandler::_init(void) {
  addThread(0);
  _timer.async_wait(boost::bind(&ConcurrencyHandler::_handler, this));
  run();
  return;
}

void ConcurrencyHandler::onEvent(EventData* eventData) {
  stack<string>* myStack;
  boost::mutex* myMut;
  if (!_terminate) {
    switch(eventData->eventType) {

      case START_EVENT:
      {
        TimerEventData *starter = (TimerEventData*)eventData;
        myStack = getEventStack(starter->threadID);
        myMut = getEventStackMutex(starter->threadID);
        myMut->lock();
        myStack->push(*(starter->timerName));
        myMut->unlock();
        break;
      }

      case STOP_EVENT:
      {
        TimerEventData *stopper = (TimerEventData*)eventData;
        myStack = getEventStack(stopper->threadID);
        myMut = getEventStackMutex(stopper->threadID);
        myMut->lock();
        myStack->pop();
        myMut->unlock();
        break;
      }

      case NEW_THREAD:
      {
        NewThreadEventData *newthread = (NewThreadEventData*)eventData;
        addThread(newthread->threadID);
        break;
      }

      case SHUTDOWN:
      {
        ShutdownEventData *shutdown = (ShutdownEventData*)eventData;
        _terminate = true;   
        outputSamples(shutdown->nodeID);
        break;
      }

      default:
      {
        break;
      }

    }
  }
  return;
}

inline stack<string>* ConcurrencyHandler::getEventStack(unsigned int tid) {
  stack<string>* tmp;
  _vectorMutex.lock();
  tmp = this->_eventStack[tid];
  _vectorMutex.unlock();
  return tmp;
}

inline boost::mutex* ConcurrencyHandler::getEventStackMutex(unsigned int tid) {
  boost::mutex* tmp;
  _vectorMutex.lock();
  tmp = this->_stackMutex[tid];
  _vectorMutex.unlock();
  return tmp;
}

inline void ConcurrencyHandler::reset(void) {
  if (!_terminate) {
    _timer.expires_at(_timer.expires_at() + boost::posix_time::microseconds(_period));
    _timer.async_wait(boost::bind(&ConcurrencyHandler::_handler, this));
  }
}

inline void ConcurrencyHandler::addThread(unsigned int tid) { 
  _vectorMutex.lock();
  while(_eventStack.size() <= tid) {
    _eventStack.push_back(new stack<string>); 
    _stackMutex.push_back(new boost::mutex);
  }
  _vectorMutex.unlock();
}

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
  } else {
    demangled = new string(timer_name);
  }
  return demangled;
}

bool sortFunctions(pair<string,int> first, pair<string,int> second) {
  if (first.second > second.second)
    return true;
  return false;
}

void ConcurrencyHandler::outputSamples(int nodeID) {
  //cout << _states.size() << " samples seen:" << endl;
  ofstream myfile;
  stringstream datname;
  datname << "concurrency." << nodeID << ".dat"; 
  myfile.open(datname.str().c_str()); 
  _functionMutex.lock();
  // limit ourselves to 5 functions.
  map<string, int> funcCount;
  // initialize the map
  for (set<string>::iterator it=_functions.begin(); it!=_functions.end(); ++it) {
    funcCount[*it] = 0;
  }
  // count all function instances
  for (unsigned int i = 0 ; i < _states.size() ; i++) {
    for (set<string>::iterator it=_functions.begin(); it!=_functions.end(); ++it) {
      if (_states[i]->find(*it) == _states[i]->end()) {
        continue;
      } else {
        funcCount[*it] = funcCount[*it] + (*(_states[i]))[*it];
      }
    }
  }
  // sort the map
  vector<pair<string,int> > myVec(funcCount.begin(), funcCount.end());
  sort(myVec.begin(),myVec.end(),&sortFunctions);
  set<string> topX;
  for (vector<pair<string, int> >::iterator it=myVec.begin(); it!=myVec.end(); ++it) {
	//if (topX.size() < 15 && (*it).first != "APEX THREAD MAIN")
	if (topX.size() < 15)
      topX.insert((*it).first);
  }

  // output the header
  for (set<string>::iterator it=_functions.begin(); it!=_functions.end(); ++it) {
	if (topX.find(*it) != topX.end()) {
	  string* tmp = demangle(*it);
      myfile << "\"" << *tmp << "\"\t";
	  delete (tmp);
	}
  }
  myfile << "\"other\"" << endl;

  unsigned int maxY = 0;
  unsigned int maxX = _states.size();
  for (unsigned int i = 0 ; i < maxX ; i++) {
    unsigned int tmpMax = 0;
	int other = 0;
    for (set<string>::iterator it=_functions.begin(); it!=_functions.end(); ++it) {
	  // this is the idle event.
	  //if (*it == "APEX THREAD MAIN")
	    //continue;
	  int value = 0;
	  // did we see this timer during this sample?
      if (_states[i]->find(*it) != _states[i]->end()) {
        value = (*(_states[i]))[*it];
      }
	  // is this timer in the top X?
	  if (topX.find(*it) == topX.end()) {
	    other = other + value;
	  } else {
        myfile << (*(_states[i]))[*it] << "\t";
        tmpMax += (*(_states[i]))[*it];
	  }
    }
    myfile << other << "\t" << endl;
    tmpMax += other;
    if (tmpMax > maxY) maxY = tmpMax;
  }
  _functionMutex.unlock();
  myfile.close();

  stringstream plotname;
  plotname << "concurrency." << nodeID << ".gnuplot";
  myfile.open(plotname.str().c_str()); 
  myfile << "set key outside bottom center invert box" << endl;
  myfile << "unset xtics" << endl;
  myfile << "set xrange[0:" << maxX << "]" << endl;
  myfile << "set yrange[0:" << maxY << "]" << endl;
  myfile << "set xlabel \"Time\"" << endl;
  myfile << "set ylabel \"Concurrency\"" << endl;
  myfile << "# Select histogram data" << endl;
  myfile << "set style data histogram" << endl;
  myfile << "# Give the bars a plain fill pattern, and draw a solid line around them." << endl;
  myfile << "set style fill solid border" << endl;
  myfile << "set style histogram rowstacked" << endl;
  myfile << "set boxwidth 1.0 relative" << endl;
  myfile << "set palette rgb 33,13,10" << endl;
  myfile << "unset colorbox" << endl;
  myfile << "plot for [COL=1:" << topX.size()+1;
  myfile << "] '" << datname.str().c_str();
  myfile << "' using COL:xticlabels(1) palette frac COL/" << topX.size()+1;
  myfile << ". title columnheader" << endl;
  myfile.close();
}

}
