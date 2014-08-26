#ifndef POLICYHANDLER_HPP
#define POLICYHANDLER_HPP

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

class PolicyHandler : public Handler, public EventListener {
private:
  void _init(void);
public:
  PolicyHandler (void);
  PolicyHandler (unsigned int period);
  ~PolicyHandler (void) { };
  void onEvent(EventData* eventData);
  void _handler(void);
};

}

#endif // POLICYHANDLER_HPP
