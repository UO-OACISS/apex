#ifndef CONCURRENCYHANDLER_HPP
#define CONCURRENCYHANDLER_HPP

#include "Handler.hpp"
#include "EventListener.hpp"
#include <stack>
#include <vector>
#include <map>
#include <set>
#include <boost/thread/mutex.hpp>

#ifdef SIGEV_THREAD_ID
#ifndef sigev_notify_thread_id
#define sigev_notify_thread_id _sigev_un._tid
#endif /* ifndef sigev_notify_thread_id */
#endif /* ifdef SIGEV_THREAD_ID */

using namespace std;

namespace apex {

class ConcurrencyHandler : public Handler, public EventListener {
private:
  void _init(void);
  // vectors and mutex
  vector<stack<string>* > _eventStack;
  boost::mutex _vectorMutex;
  vector<boost::mutex* > _stackMutex;
  // periodic samples of stack top states
  vector<map<string, unsigned int>* > _states;
  // functions and mutex
  set<string> _functions;
  boost::mutex _functionMutex;
  int _option;
public:
  ConcurrencyHandler (void);
  ConcurrencyHandler (char *option);
  ConcurrencyHandler (unsigned int period);
  ConcurrencyHandler (unsigned int period, char* option);
  ~ConcurrencyHandler (void) { };
  void onEvent(EventData* eventData);
  void _handler(void);
  stack<string>* getEventStack(unsigned int tid);
  boost::mutex* getEventStackMutex(unsigned int tid);
  void addThread(unsigned int tid) ;
  void reset(void);
  void outputSamples(int nodeID);
};

}

#endif // CONCURRENCYHANDLER_HPP
