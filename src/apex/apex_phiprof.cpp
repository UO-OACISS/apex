#include <map>
#include <mpi.h>
#include <mutex>
#include <stack>
#include "apex.hpp"
#include <phiprof.hpp>

//#warning compiling phiprof

/* Correspondance between labels and ids */

std::mutex mut;

std::map<std::string, int> timers_labels;
std::map<int, std::string> timers_ids;

int insertOrGet( const std::string &label ){
    int id;
    mut.lock();
    auto it = timers_labels.find( label );
    if( it == timers_labels.end() ){
        id = timers_labels.size();
        timers_labels[ label ] = id;
        timers_ids[ id ] = label;
    } else {
        id =  it->second;
    }    
    mut.unlock();
    return id;
}


/* Simple initialization. Returns true if started succesfully */

bool phiprof::initialize(){
    return true;
}

/* Initialize a timer, with a particular label   
 *
 * Initialize a timer. This enables one to define groups, and to use
 * the return id value for more efficient starts/stops in tight
 * loops. If this function is called for an existing timer it will
 * simply just return the id.
 */

int phiprof::initializeTimer(const std::string &label, const std::vector<std::string> &groups){
    return insertOrGet( label );
}

int phiprof::initializeTimer(const std::string &label){
    return insertOrGet( label );
}

int phiprof::initializeTimer(const std::string &label,const std::string &group1) {
    return insertOrGet( label );
}

int phiprof::initializeTimer(const std::string &label,const std::string &group1,const std::string &group2){
    return insertOrGet( label );
}

int phiprof::initializeTimer(const std::string &label,const std::string &group1,const std::string &group2,const std::string &group3){
    return insertOrGet( label );
}

/* Get id number of an existing timer that is a child of the currently
 * active one */

int phiprof::getChildId(const std::string &label){
    /* TODO */
    return -1;
}

/* Start-stop timers */

static thread_local std::stack<std::shared_ptr<apex::task_wrapper> > my_stack;


bool phiprof::start(const std::string &label){
    auto t = apex::new_task( label );
    apex::start( t );
    my_stack.push(t);
    return true;
}

bool phiprof::start(int id){
    auto label = timers_ids[ id ];
    auto t = apex::new_task( label );
    apex::start( t );
    my_stack.push(t);
    return true;
}

bool phiprof::stop (const std::string &label, double workUnits, const std::string &workUnitLabel){
    auto t = my_stack.top();
    apex::stop(t);
    my_stack.pop();
    return true;
}

bool phiprof::stop (int id, double workUnits, const std::string &workUnitLabel){
    auto t = my_stack.top();
    apex::stop(t);
    my_stack.pop();
    return true;
}

bool phiprof::stop (int id){
    auto t = my_stack.top();
    apex::stop(t);
    my_stack.pop();
    return true;
}

bool phiprof::print(MPI_Comm comm, std::string fileNamePrefix){
    return true;
}

