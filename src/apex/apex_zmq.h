
namespace apex {

class ZmqSession {
public:
    ZmqSession(int commrank);
    ~ZmqSession(void);
    void runServer(void);
    void runClient(void);
private:
    int global_rank;
    int local_rank;
    const char * ipc_location{"ipc:///tmp/apex.feeds.0"};
};

} // namespace apex