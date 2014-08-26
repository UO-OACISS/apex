#ifndef TAUHANDLER_HPP
#define TAUHANDLER_HPP

#include "EventListener.hpp"

using namespace std;

namespace apex {

class TauListener : public EventListener {
private:
  void _init(void);
  bool _terminate;
public:
  TauListener (void);
  ~TauListener (void) { };
  void onEvent(EventData* eventData);
};

}

#endif // TAUHANDLER_HPP
