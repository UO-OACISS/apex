#include <string>
#include <iostream>
#include <thread>
#include <zmq.hpp>
#include "apex_zmq.h"

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
        return false;
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

ZmqSession::~ZmqSession(void) {}

void ZmqSession::runServer(void) {
    // initialize the zmq context with a single IO thread
    zmq::context_t context{1};

    // construct a REP (reply) socket and bind to interface
    zmq::socket_t socket{context, zmq::socket_type::rep};
    //socket.bind("tcp://*:5555");
    socket.bind(ipc_location);

    // prepare some static data for responses
    std::string data{"World from "};
    data += std::to_string(global_rank);

    for (int i = 0; i < 3; i++)
    {
        zmq::message_t request;

        // receive a request from client
        zmq::recv_result_t ret(socket.recv(request, zmq::recv_flags::none));
        if (ret.has_value() && (EAGAIN == ret.value()))
        {
            // msocket had nothing to read and recv() timed out
            std::cout << "Not Received " << request.to_string() << std::endl;
        }
        std::cout << "Received " << request.to_string() << std::endl;

        // send the reply to the client
        socket.send(zmq::buffer(data), zmq::send_flags::none);
    }
}

void ZmqSession::runClient(void) {
    // initialize the zmq context with a single IO thread
    zmq::context_t context{1};

    // construct a REQ (request) socket and connect to interface
    zmq::socket_t socket{context, zmq::socket_type::req};
    //socket.connect("tcp://localhost:5555");
    socket.connect(ipc_location);

    // set up some static data to send
    std::string data{"Hello from "};
    data += std::to_string(global_rank);

    for (auto request_num = 0; request_num < 1; ++request_num)
    {
        // send the request message
        std::cout << "Sending Hello from " << global_rank << "..." << std::endl;
        socket.send(zmq::buffer(data), zmq::send_flags::none);

        // wait for reply from server
        zmq::message_t reply{};
        zmq::recv_result_t ret(socket.recv(reply, zmq::recv_flags::none));
        if (ret.has_value() && (EAGAIN == ret.value()))
        {
            // msocket had nothing to read and recv() timed out
            std::cout << "Not Received " << reply.to_string() << std::endl;
        }

        std::cout << "Received " << reply.to_string();
        std::cout << std::endl;
    }
}

} // namespace apex
