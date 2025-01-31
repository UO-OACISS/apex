#include "apex_zmq.h"
#include <string>
#include <iostream>
#include <chrono>
#include <thread>

namespace apex {

int test_for_MPI_local_rank(int commrank) {
    /* Some configurations might use MPI without telling ZeroSum - they can
     * call apex::init() with a rank of 0 and size of 1 even though
     * they are running in an MPI application.  For that reason, we double
     * check to make sure that we aren't in an MPI execution by checking
     * for some common environment variables. */
    // PMI, MPICH, Cray, Intel, MVAPICH2...
    const auto helper = [&](const char * variable) {
        const char * tmpvar = getenv(variable);
        if (tmpvar != NULL) {
            commrank = atol(tmpvar);
            // printf("Changing PMI rank to %lu\n", commrank);
            return true;
        }
    };
    if (helper("PMI_LOCAL_RANK")) return commrank;
    if (helper("PALS_LOCAL_RANKID")) return commrank;
    if (helper("OMPI_COMM_WORLD_LOCAL_RANK")) return commrank;
    if (helper("SLURM_LOCALID")) return commrank;
    return commrank;
}

ZmqSession::ZmqSession(int commrank) :
    global_rank(commrank),
    local_rank(test_for_MPI_local_rank(commrank))
{
    if (local_rank == 0) {
        runServer();
    } else {
        runClient();
    }
}

void ZmqSession::runServer() {
    using namespace std::chrono_literals;

    // initialize the zmq context with a single IO thread
    zmq::context_t context{1};

    // construct a REP (reply) socket and bind to interface
    zmq::socket_t socket{context, zmq::socket_type::rep};
    //socket.bind("tcp://*:5555");
    socket.bind("ipc:///tmp/apex/feeds/0");

    // prepare some static data for responses
    const std::string data{"World"};

    for (;;)
    {
        zmq::message_t request;

        // receive a request from client
        socket.recv(request, zmq::recv_flags::none);
        std::cout << "Received " << request.to_string() << std::endl;

        // simulate work
        std::this_thread::sleep_for(1s);

        // send the reply to the client
        socket.send(zmq::buffer(data), zmq::send_flags::none);
    }
}

void ZmqSession::runClient() {
    // initialize the zmq context with a single IO thread
    zmq::context_t context{1};

    // construct a REQ (request) socket and connect to interface
    zmq::socket_t socket{context, zmq::socket_type::req};
    //socket.connect("tcp://localhost:5555");
    socket.bind("ipc:///tmp/apex/feeds/0");

    // set up some static data to send
    const std::string data{"Hello"};

    for (auto request_num = 0; request_num < 10; ++request_num)
    {
        // send the request message
        std::cout << "Sending Hello " << request_num << "..." << std::endl;
        socket.send(zmq::buffer(data), zmq::send_flags::none);

        // wait for reply from server
        zmq::message_t reply{};
        socket.recv(reply, zmq::recv_flags::none);

        std::cout << "Received " << reply.to_string();
        std::cout << " (" << request_num << ")";
        std::cout << std::endl;
    }
}

} // namespace apex
