#ifndef RCRBLACKBOARD_H
#define RCRBLACKBOARD_H

#include <stdint.h> // for int64_t
#include <stdio.h>  // for printf
#include <stdlib.h>  // for malloc - temp until share memory region allocated
#include <sys/mman.h> // for mmap
#include <fcntl.h>   // for mmap #defs
#include <unistd.h>   // for ftruncate

#include "RCR.bb.h"

using namespace std;

class RCRblackboard {
private:
  struct _RCRBlackboard *bb; // pointer to _RCRBlackboard in shared memory
  int64_t allocationHighWater;
  void * bbMem;

public:
  RCRblackboard();  // allocates shared memory and creates blackboard
  RCRblackboard(const RCRblackboard &bb);  // allocates shared memory and creates blackboard
  ~RCRblackboard();

  // Punting on protection -- trusing readers to only read -- eventually use the OS
  //   page protection to prevent writing by readers

  // return address of RCRmeter for later direct use
  volatile int64_t  *getThreadMeter(enum rcrMeterType id, int64_t node, int64_t socket,
			  int64_t core, int64_t thread); // get a specific thread meter
  volatile int64_t  *getCoreMeter(enum rcrMeterType id, int64_t node, int64_t socket,
			int64_t core); // tet a specific core meter
  volatile int64_t  *getSocketMeter(enum rcrMeterType id, int64_t node, int64_t socket
			   ); // get a specific socket meter
  volatile int64_t  *getNodeMeter(enum rcrMeterType id, int64_t node); // get a specific node meter 
  volatile int64_t  *getSystemMeter(enum rcrMeterType id); // get a specific system meter 
 
  int64_t getNumOfNodes();
  int64_t getNumOfSockets();
  int64_t getNumOfCores();
  int64_t getNumOfThreads();
  char*   getNodeName();
  int64_t setSystemSize(int64_t initOffset);
  int64_t setNodeSize(int64_t initOffset);
  int64_t setSocketSize(int64_t initOffset);
  int64_t setCoreSize(int64_t initOffset);
  int64_t setThreadSize(int64_t initOffset);

  int64_t setNodeOffset(int64_t nodeID);
  int64_t setSocketOffset(int64_t nodeID, int64_t socketID);
  int64_t setCoreOffset(int64_t nodeID, int64_t socketID, int64_t coreID);
  int64_t setThreadOffset(int64_t nodeID, int64_t socketID, int64_t coreID, int64_t threadID);

  // assign a meter for each thread -- returns true if all assigned
  bool assignSystemMeter(enum rcrMeterType id);
  bool assignNodeMeter(int64_t nodeId, enum rcrMeterType id);
  bool assignSocketMeter(int64_t nodeId, int64_t socketId, enum rcrMeterType id);
  bool assignCoreMeter(int64_t nodeId, int64_t socketId,
		       int64_t coreId, enum rcrMeterType id);
  bool assignThreadMeter(int64_t nodeId, int64_t socketId,
			 int64_t coreId, int64_t threadId, enum rcrMeterType id);

  // Initial implementation builds a homogeneous system -- need a heterogeneous build
  //   sometime in the future (probably a bottom-up build rather than the current
  //   top-down version

  int64_t buildThread(int64_t threadId, int64_t numThreadMeters);
  int64_t buildCore(int64_t c, int64_t coreMeters, int64_t numThread);
  int64_t buildSocket(int64_t s, int64_t socketMeters, int64_t numCore);
  int64_t buildNode(int64_t n, int64_t nodeMeters, int64_t numSocket);
  int64_t buildSystem(int64_t numBBMeters, int64_t numNodes);

  // build a homogeneous blackboard structure in shared memory
  bool buildSharedMemoryBlackBoard(int64_t systemMeters,
				   int64_t numNode, int64_t nodeMeters,
				   int64_t numSocket, int64_t socketMeters,
				   int64_t numCore, int64_t coreMeters,
				   int64_t numThread, int64_t threadMeters
				   ); 

  // read existing blackboard 
  bool initBlackBoard();


private:

  char * getMacAddress();
  char * base();
  
  int64_t allocateSharedMemory(int64_t size);
  int64_t getCurOffset();

  struct _RCRNode   * getNode(int nodeId);
  struct _RCRSocket * getSocket(int nodeId, int socketId);
  struct _RCRCore   * getCore(int nodeId, int socketId, int coreId);
  struct _RCRThread * getThread(int nodeId, int socketId, int coreId, int threadId);

};

#endif
