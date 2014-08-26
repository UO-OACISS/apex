#ifndef APEX_HANDLER_H
#define APEX_HANDLER_H

#include <string>
#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;

namespace apex {

class Handler
{
private:
  static boost::asio::io_service _io;
  static const unsigned int defaultPeriod = 100000;
  void _threadfunc(void) {
    _io.run();
  };
protected:
  unsigned int _period;
  bool _terminate;
  boost::asio::deadline_timer _timer;
  boost::thread* _timerThread;
  void reset(void) {
      _timer.expires_at(_timer.expires_at() + boost::posix_time::microseconds(_period));
      _timer.async_wait(boost::bind(&Handler::_handler, this));
  };
  void run(void) {
    _timerThread = new boost::thread(&Handler::_threadfunc, this);
  };
public:
  Handler() : _period(defaultPeriod), _terminate(false), _timer(_io, boost::posix_time::microseconds(_period)) { }
  Handler(unsigned int period) : _period(period), _terminate(false), _timer(_io, boost::posix_time::microseconds(_period)) { }
  // virtual destructor
  virtual ~Handler() {};
  // all methods in the interface that a handler has to override
  virtual void _handler(void) {
      cout << "Default handler" << endl;
      this->reset();
  };
};

}

#endif // APEX_HANDLER_H
