#include <zmq.hpp>

namespace apex {

class ZmqSession {
public:
    ZmqSession(int commrank);
    ~ZmqSession();
    void runServer();
    void runClient();
private:
    int global_rank;
    int local_rank;
};

} // namespace apex