#include <stdint.h> // for int64_t
#include <stdio.h>  // for printf
#include <stdlib.h>  // for malloc - temp until share memory region allocated
#include <sys/mman.h> // for mmap
#include <fcntl.h>   // for mmap #defs
#include <unistd.h>   // for ftruncate
#include <iostream>

#include "RCR.bb.h"
#include "RCRblackboard.hpp"

//
//
// TODO 
//   error messages in getNode et al functions
//
//



//
// RCRblackboard -- RCRdaemon subsection
//
// writer -- allocates its required counters and updates the specified addresses
//           expect an array of address to update will be kept and the lookup will only be
//           part of initialization
// reader -- locates the desired counters and when desired reads the most recent valuses
//           no protection from reader writing at the moment -- be careful
//           expect that meter locations will be saved and lookup will only occur during
//           initialization


using namespace std;

// default/empty constructor and destructor routines  -- fill in if needed -- akp
RCRblackboard::RCRblackboard(const RCRblackboard &bb)
{
}

RCRblackboard::RCRblackboard()
{
}

RCRblackboard::~RCRblackboard()
{
}

/*****************************************
 *  private internal routines
 ****************************************/

// get address of shared memory region -- used as a base to compute locations using offsets
//   thoughtout the blackboard 

char * RCRblackboard::base(){
  return (char*)bbMem;
}

// allocate next block of shared memory -- track highwater mark

int64_t RCRblackboard::allocateSharedMemory(int64_t size){
  int64_t ret = allocationHighWater;
  if(size == 0) size = 8;// put blank word in if no space required to prevent field overlaps
  if(size + allocationHighWater > MAX_RCRFILE_SIZE){
    std::cerr << " Out of shared memory" << std::endl;
    exit(1);
  }
  if(ret == 0) { // no allocation invent some memory

    int fd = shm_open(RCRFILE_NAME, O_RDWR, 0);  // TODO should be #def or execution time definable
    if(fd == -1) {
        perror("shm_open");
        exit(1);
    }

    ftruncate(fd, MAX_RCRFILE_SIZE);
    bbMem = mmap(NULL, MAX_RCRFILE_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

  }
  allocationHighWater += size;
  return ret;
}

// get current allocation offset -- next allocation starting offset

int64_t RCRblackboard::getCurOffset(){
  return allocationHighWater;
}

// rountines to aquire the blackboard internal stuctures given location in system tree

struct _RCRNode * RCRblackboard::getNode(int nodeNum)
{
  int64_t *nodeOffset = (int64_t *)(base() + bb->nodeOffset);
  struct _RCRNode * node = (struct _RCRNode*)(base() + *(nodeOffset + nodeNum));
  return node;
}

struct _RCRSocket * RCRblackboard::getSocket(int nodeNum,
					     int socketNum)
{
  int64_t *socketOffset = (int64_t *)(base() + getNode(nodeNum)->socketOffset);
  struct _RCRSocket * socket = (struct _RCRSocket*)(base() + *(socketOffset+socketNum));
  return socket;
}

struct _RCRCore * RCRblackboard::getCore(int nodeNum,
					 int socketNum,
					 int coreNum)
{
  int64_t *coreOffset = (int64_t *)(base() + getSocket(nodeNum, socketNum)->coreOffset);
  struct _RCRCore * core = (struct _RCRCore*)(base() + *(coreOffset + coreNum));
  return core;
}

struct _RCRThread * RCRblackboard::getThread(int nodeNum,
					     int socketNum,
					     int coreNum,
					     int threadNum)
{
  int64_t *threadOffset = (int64_t *)(base() + getCore(nodeNum, socketNum, coreNum)->threadOffset);
  struct _RCRThread * thread = (struct _RCRThread*)(base() + *(threadOffset + threadNum));
  return thread;
}

/*******************************
 * public interface functions
 *******************************/

// Get number of nodes/sockets/cores/threads in current blackboard system
//assumes # sockets per node all the same

/********************************************************************************/
/*********** ALL OF THESE SIZE FUNCTIONS ASSUME HOMOGENEOUS SYSTEM **************/
/********************************************************************************/

int64_t RCRblackboard::getNumOfNodes()
{
  return bb->numNodes;
}

int64_t RCRblackboard::getNumOfSockets()
{
  struct _RCRNode * node = getNode(0);
  return node->numSockets;
}

int64_t RCRblackboard::getNumOfCores()
{
  struct _RCRSocket * socket = getSocket(0,0);
  return socket->numCores;
}

int64_t RCRblackboard::getNumOfThreads()
{
  struct _RCRCore * core = getCore(0,0,0);
  return core->numThreads;
}


/*******************************
 * routines to locate a particular meter -- need to know location and desired type
 *   returns ponter to counter/meter that can be used for the life of the program
 ******************************/
  
// get a specific system meter 
volatile int64_t *RCRblackboard::getSystemMeter(enum rcrMeterType id)
{
  int64_t i = 0; // current index being checked

  // location of first meter
  struct _RCRMeterValue *cur  = (struct _RCRMeterValue*)(base() + bb->bbMeterOffset);

  while (i < bb->numBBMeters) {  // check each until all meters examined
    if(id == cur->meterID) return &(cur->data.iValue); // return address of counter/meter
    cur++;
    i++;
  }

  return NULL; // ID not found 
}

// get a specific node meter 
volatile int64_t *RCRblackboard::getNodeMeter(enum rcrMeterType id, int64_t nodeId)
{
  int64_t i = 0; // current index being checked

  struct _RCRNode * node = getNode(nodeId);

  // location of first meter
  struct _RCRMeterValue *cur  = (struct _RCRMeterValue*)(base() + node->nodeMeterOffset);

  while (i < node->numNodeMeters) {  // check each until all meters examined
    if(id == cur->meterID) return &(cur->data.iValue); // return address of counter/meter
    cur++;
    i++;
  }

  return NULL; // ID not found 
}

// get a specific socket meter 
volatile int64_t *RCRblackboard::getSocketMeter(enum rcrMeterType id,
						int64_t nodeId, 
						int64_t socketId)
{
  int64_t i = 0; // current index being checked

  struct _RCRSocket * socket = getSocket(nodeId, socketId);

  // location of first meter
  struct _RCRMeterValue *cur  = (struct _RCRMeterValue*)(base() + socket->socketMeterOffset);

  while (i < socket->numSocketMeters) {  // check each until all meters examined
    if(id == cur->meterID) return &(cur->data.iValue); // return address of counter/meter
    cur++;
    i++;
  }

  return NULL; // ID not found 
}


// get a specific core meter 
volatile int64_t *RCRblackboard::getCoreMeter(enum rcrMeterType id,
					      int64_t nodeId,
					      int64_t socketId,
					      int64_t coreId)
{
  int64_t i = 0; // current index being checked

  struct _RCRCore * core = getCore(nodeId, socketId, coreId);

  // location of first meter
  struct _RCRMeterValue *cur  = (struct _RCRMeterValue*)(base() + core->coreMeterOffset);

  while (i < core->numCoreMeters) {  // check each until all meters examined
    if(id == cur->meterID) return &(cur->data.iValue); // return address of counter/meter
    cur++;
    i++;
  }

  return NULL; // ID not found 
}

// get a specific thread meter 
volatile int64_t *RCRblackboard::getThreadMeter(enum rcrMeterType id, 
						int64_t nodeId,
						int64_t socketId,
						int64_t coreId,
						int64_t threadId)
{
  int64_t i = 0; // current index being checked

  struct _RCRThread * thread = getThread(nodeId, socketId, coreId, threadId);

  // location of first meter
  struct _RCRMeterValue *cur  = (struct _RCRMeterValue*)(base() + thread->threadMeterOffset);

  while (i < thread->numThreadMeters) {  // check each until all meters examined
    if(id == cur->meterID) return &(cur->data.iValue); // return address of counter/meter
    cur++;
    i++;
  }

  return NULL; // ID not found 
}

/******************************
 * assign a meter to be collected
 *   if counter/meter added returns true
 *   else returns false (all available counter/meters allocated)
 *****************************/

bool RCRblackboard::assignSystemMeter(enum rcrMeterType id)
{
  int64_t i = 0; // current index being checked

  struct _RCRMeterValue *cur = (struct _RCRMeterValue *)(base() + bb->bbMeterOffset); 

  while (i < bb->numBBMeters) {  // check each until all meters examined

    if(cur->meterID == -1) {
      cur->meterID = id;
      cur->data.iValue=0;
      return true;
    }
    cur++;
    i++;
  }

  std::cerr << "no room to assign System meter" << std::endl;
  return false; // no room for assigned meter

};

bool RCRblackboard::assignNodeMeter(int64_t nodeId, enum rcrMeterType id)
{
  int64_t i = 0; // current index being checked

  struct _RCRNode * node = getNode(nodeId);
  struct _RCRMeterValue *cur = (struct _RCRMeterValue *)(base() + node->nodeMeterOffset); 

  while (i < node->numNodeMeters) {  // check each until all meters examined

    if(cur->meterID == -1) {
      cur->meterID = id;
      cur->data.iValue=0;
      return true;
    }
    cur++;
    i++;
  }

  std::cerr << "no room to assign Node meter on Node" << nodeId  << std::endl;
  return false; // no room for assigned meter
};

bool RCRblackboard::assignSocketMeter(int64_t nodeId, int64_t socketId, enum rcrMeterType id){

  int64_t i = 0; // current index being checked

  struct _RCRSocket * socket = getSocket(nodeId, socketId);
  struct _RCRMeterValue *cur = (struct _RCRMeterValue *)(base() + socket->socketMeterOffset); 

  while (i < socket->numSocketMeters) {  // check each until all meters examined

    if(cur->meterID == -1) {
      cur->meterID = id;
      cur->data.iValue=0;
      return true;
    }
    cur++;
    i++;
  }

  std::cerr << "no room to assign Socket meter on Node " << nodeId << " Socket " << socketId << std::endl;
  return false; // no room for assigned meter
};

bool RCRblackboard::assignCoreMeter(int64_t nodeId,
				    int64_t socketId,
				    int64_t coreId,
				    enum rcrMeterType id)
{
  int64_t i = 0; // current index being checked

  struct _RCRCore * core = getCore(nodeId, socketId, coreId);
  struct _RCRMeterValue *cur = (struct _RCRMeterValue *)(base() + core->coreMeterOffset); 

  while (i < core->numCoreMeters) {  // check each until all meters examined
    
    if(cur->meterID == -1) {
      cur->meterID = id;
      cur->data.iValue=0;
      return true;
    }
    cur++;
    i++;
  }

  std::cerr << "no room to assign Core meter on Node " << nodeId << " Socket " 
	    << socketId << " Core " << coreId <<std::endl;
  return false; // no room for assigned meter
};

bool RCRblackboard::assignThreadMeter(int64_t nodeId,
				      int64_t socketId,
				      int64_t coreId,
				      int64_t threadId,
				      enum rcrMeterType id)
{
  int64_t i = 0; // current index being checked

  struct _RCRThread *thread = getThread(nodeId, socketId, coreId, threadId);
  struct _RCRMeterValue *cur = (struct _RCRMeterValue *)(base() + thread->threadMeterOffset);

  while (i < thread->numThreadMeters) {  // check each until all meters examined

    if(cur->meterID == -1) {
      cur->meterID = id;
      cur->data.iValue=0;
      return true;
    }
    cur++;
    i++;
  }

  std::cerr << "no room to assign Core meter on Node " << nodeId << " Socket " 
	    << socketId << " Core " << coreId << std::endl;
  return false; // no room for assigned meter
};


/**********************************
 *  Build pieces of the system during initialization
 *********************************/

int64_t RCRblackboard::buildSystem(int64_t numBBMeters, int64_t numNodes)
{
  int64_t i;
  int64_t offset = allocateSharedMemory(sizeof(struct _RCRBlackboard));

  bb = (struct _RCRBlackboard*)(base() + offset);
  bb->md.bbVersionNumber = RCRBlackBoardVersionNumber; 
  bb->numBBMeters = numBBMeters;
  bb->bbMeterOffset = allocateSharedMemory(sizeof(_RCRMeterValue) * numBBMeters);

  for (i = 0; i < numBBMeters; i ++){
    struct _RCROffset *meterOffset = (struct _RCROffset*)(base() + bb->bbMeterOffset);
    struct _RCRMeterValue *cur = (struct _RCRMeterValue *)(base() + meterOffset->offset[i]); 
    cur->meterID = -1;   // allocated -- free
  }
  bb->numNodes = numNodes;
  bb->nodeOffset = allocateSharedMemory(sizeof(int64_t) * (numNodes+1)); // 1 for size field
  // AKP -- is addition for size field correct??
  return offset;
}

int64_t RCRblackboard::buildNode(int64_t nodeId, int64_t numNodeMeters, int64_t numSockets)
{
  int i;
  int64_t offset = allocateSharedMemory(sizeof(struct _RCRNode));

  struct _RCRNode* node = (struct _RCRNode*)(base() + offset);
  node->md.nodeType = 0; 
  node->md.nodeNumber = nodeId; 
  node->md.nodeTreeSize = 0;
  node->numNodeMeters = numNodeMeters;
  node->nodeMeterOffset = allocateSharedMemory(sizeof(_RCRMeterValue) * numNodeMeters);

  for (i = 0; i < numNodeMeters; i ++){
    struct _RCRMeterValue *cur = (struct _RCRMeterValue *)(base() + node->nodeMeterOffset+(sizeof(struct _RCRMeterValue)*i)); // use offset to find
    cur->meterID = -1;  // allocated -- free
  }
  node->numSockets = numSockets;
  node->socketOffset = allocateSharedMemory(sizeof(_RCRMeterValue) * (numSockets+1));
  // 1 for size field  -- AKP -- is addition for size field correct??
  return offset;
};  

int64_t RCRblackboard::buildSocket(int64_t socketId, int64_t numSocketMeters, int64_t numCores){
  int i;
  int64_t offset = allocateSharedMemory(sizeof(struct _RCRSocket));

  struct _RCRSocket* socket = (struct _RCRSocket*)(base() + offset);
  socket->md.socketType = 0;
  socket->md.socketNumber = socketId;
  socket->md.socketTreeSize = 0;
  socket->numSocketMeters = numSocketMeters;
  socket->socketMeterOffset = allocateSharedMemory(sizeof(_RCRMeterValue) * numSocketMeters);

  for (i = 0; i < numSocketMeters; i ++){
    struct _RCRMeterValue *cur = (struct _RCRMeterValue *)(base() + socket->socketMeterOffset+(sizeof(struct _RCRMeterValue)*i)); // use offset to find
    cur->meterID = -1;     // allocated -- free
  }

  socket->numCores = numCores;
  socket->coreOffset = allocateSharedMemory(sizeof(_RCRMeterValue) * (numCores+1));
  // 1 for size field  -- AKP -- is addition for size field correct??
  return offset;
};  

int64_t RCRblackboard::buildCore(int64_t coreId, int64_t numCoreMeters, int64_t numThreads)
{
  int64_t i;
  int64_t offset = allocateSharedMemory(sizeof(struct _RCRCore));

  struct _RCRCore* core = (struct _RCRCore*)(base() + offset);
  core->md.coreType = 0; 
  core->md.coreNumber = coreId; 
  core->md.coreTreeSize = 0; 
  core->numCoreMeters = numCoreMeters;
  core->coreMeterOffset = allocateSharedMemory(sizeof(_RCRMeterValue) * numCoreMeters);

  for (i = 0; i < numCoreMeters; i ++){
    struct _RCRMeterValue *cur = (struct _RCRMeterValue *)(base() + core->coreMeterOffset+(sizeof(struct _RCRMeterValue)*i)); // use offset to find
    cur->meterID = -1;     // allocated -- free
  }

  core->numThreads = numThreads;
  core->threadOffset = allocateSharedMemory(sizeof(_RCRMeterValue) * (numThreads+1));
  // 1 for size field  -- AKP -- is addition for size field correct??
  return offset;
};  

int64_t RCRblackboard::buildThread(int64_t threadId, int64_t numThreadMeters)
{
  int64_t i;
  int64_t offset = allocateSharedMemory(sizeof(struct _RCRThread));

  struct _RCRThread * thread = (struct _RCRThread*)(base() + offset);
  thread->md.threadType = 0; 
  thread->md.threadNumber = threadId; 
  thread->md.threadTreeSize = 0; 
  thread->numThreadMeters = numThreadMeters;
  thread->threadMeterOffset = allocateSharedMemory(sizeof(_RCRMeterValue) * numThreadMeters);

  for (i = 0; i < numThreadMeters; i ++){
    struct _RCRMeterValue *cur = (struct _RCRMeterValue *)(base() + thread->threadMeterOffset+(sizeof(struct _RCRMeterValue)*i)); // use offset to find
    cur->meterID = -1;     // allocated -- free
  }
  return offset;
};

/***********************************
 * functions to set my tree size and offset to children after allocation 
 **********************************/

/* for each System allocation */
int64_t RCRblackboard::setSystemSize(int64_t initOffset)
{
  int64_t curOffset = getCurOffset();
  struct _RCRBlackboard * bb = (struct _RCRBlackboard*)(base() + initOffset);
  bb->md.bbTreeSize = curOffset - initOffset; // assign now that the rest is written
  return 0;
};

/* for each Node allocation */
int64_t RCRblackboard::setNodeSize(int64_t initOffset)
{
  int64_t curOffset = getCurOffset();
  struct _RCRNode * node = (struct _RCRNode*)(base() + initOffset);
  node->md.nodeTreeSize = curOffset - initOffset; // assign now that the rest is written
  return 0;
};

int64_t RCRblackboard::setNodeOffset(int64_t nodeId)
{
  int64_t *nodeOffset = (int64_t *)(base() + bb->nodeOffset + (sizeof(int64_t)*nodeId));
  *nodeOffset = getCurOffset();  
  return 0;
};

/* for each socket allocation */
int64_t RCRblackboard::setSocketSize(int64_t initOffset)
{
  int64_t curOffset = getCurOffset();
  struct _RCRSocket * sock = (struct _RCRSocket*)(base() + initOffset);
  sock->md.socketTreeSize = curOffset - initOffset; // assign now that the rest is written
  return 0;
};

int64_t RCRblackboard::setSocketOffset(int64_t nodeId, int64_t socketId)
{
  struct _RCRNode * node = getNode(nodeId);
  int64_t *socketOffset = (int64_t *)(base() + node->socketOffset +(sizeof(int64_t)*socketId));
  *socketOffset = getCurOffset();  
  return 0;
};

/* for each core allocatoin */
int64_t RCRblackboard::setCoreSize(int64_t initOffset)
{
  int64_t curOffset = getCurOffset();
  struct _RCRCore * core = (struct _RCRCore*)(base() + initOffset);
  core->md.coreTreeSize = curOffset - initOffset; // assign now that the rest is written
  return 0;
};

int64_t RCRblackboard::setCoreOffset(int64_t nodeId, int64_t socketId, int64_t coreId){
  struct _RCRSocket * socket = getSocket(nodeId, socketId);
  int64_t *coreOffset = (int64_t *)(base() + socket->coreOffset +(sizeof(int64_t)*coreId));
  *coreOffset = getCurOffset();  
  return 0;
};

/* for each thread allocation */
int64_t RCRblackboard::setThreadSize(int64_t initOffset)
{
  int64_t curOffset = getCurOffset();
  struct _RCRThread * thread = (struct _RCRThread*)(base() + initOffset);
  thread->md.threadTreeSize = curOffset - initOffset; // assign now that the rest is written
  return 0;
};

int64_t RCRblackboard::setThreadOffset(int64_t nodeId, 
				       int64_t socketId, 
				       int64_t coreId, 
				       int64_t threadId)
{
  struct _RCRCore * core = getCore(nodeId, socketId, coreId);
  int64_t *threadOffset = (int64_t *)(base() + core->threadOffset +(sizeof(int64_t)*threadId));
  *threadOffset = getCurOffset();  // set thread offset inside core
  return 0;
};



/******************************
 * Build blackboard
 *****************************/

/*********  currently build homogeneous system  ****************/

bool RCRblackboard::buildSharedMemoryBlackBoard(int64_t systemMeters,
						int64_t numNode, int64_t nodeMeters,
						int64_t numSocket, int64_t socketMeters,
						int64_t numCore, int64_t coreMeters,
						int64_t numThread, int64_t threadMeters
						){

  bool ret = true;
  allocationHighWater = 0; // overwrites previous blackboard -- should be elsewhere?
  
  int64_t systemOffset = buildSystem(systemMeters, numNode);
  
  int n = 0;
  for (n = 0; n < numNode; n++) {
    setNodeOffset(n);
    int64_t nodeOffset = buildNode(n, nodeMeters, numSocket);
    int s = 0;
    for (s = 0; s < numSocket; s++) {
      setSocketOffset(n,s);
      int64_t sockOffset = buildSocket(s, socketMeters, numCore);
      int c = 0;
      for (c = 0; c < numCore; c++) {
	setCoreOffset(n,s,c);
	int64_t coreOffset = buildCore(c, coreMeters, numThread);
	int t = 0;
	for (t = 0; t < numThread; t++) {
	  setThreadOffset(n,s,c,t);
	  int64_t threadOffset = buildThread(t,threadMeters);
	  setThreadSize (threadOffset);
	}
	setCoreSize (coreOffset);
      }
      setSocketSize (sockOffset);
    }
    setNodeSize (nodeOffset);
  }
  setSystemSize (systemOffset);
  
  std::cout << "final offset " << getCurOffset() << std::endl;
  return ret;
}


/******************************
 *  reads current blackboard file and sets it up for access 
 *****************************/

bool RCRblackboard::initBlackBoard(){
  
  int fd = shm_open(RCRFILE_NAME, O_RDONLY, 0);

  if(fd == -1) {
    perror("shm_open ");
    return false;
  }
  
  ftruncate(fd, MAX_RCRFILE_SIZE);
  bbMem = mmap(NULL, MAX_RCRFILE_SIZE, PROT_READ, MAP_SHARED, fd, 0);
  bb = (struct _RCRBlackboard*)bbMem;

  return true;
}
