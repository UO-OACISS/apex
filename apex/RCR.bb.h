#ifndef RENCI_RCRTOOL_BLACKBOARD_H
#define RENCI_RCRTOOL_BLACKBOARD_H


/* In Memory layout

blackboard struct
 system meters
 node 1
 node 1 meters offset array
 node 1 socket offset array
 node 1 meters
    node 1 socket 1
    node 1 socket 1 meters offset array
    node 1 socket 1 core offset array
    node 1 socket 1 meters
         node 1 socket 1 core 1
         node 1 socket 1 core 1 meters offset array
         node 1 socket 1 core 1 meters
         node 1 socket 1 core 2
         node 1 socket 1 core 2 meters offset array
         node 1 socket 1 core 2 meters
         node 1 socket 1 core 3
         node 1 socket 1 core 3 meters offset array
         node 1 socket 1 core 3 meters
         node 1 socket 1 core 4
         node 1 socket 1 core 4 meters offset array
         node 1 socket 1 core 4 meters
    node 1 socket 2
    node 1 socket 2 meters offset array
    node 1 socket 2 core offset array
    node 1 socket 2 meters
         node 1 socket 2 core 1
         node 1 socket 2 core 1 meters offset array
         node 1 socket 2 core 1 meters
         node 1 socket 2 core 2
         node 1 socket 2 core 2 meters offset array
         node 1 socket 2 core 2 meters
         node 1 socket 2 core 3
         node 1 socket 2 core 3 meters offset array
         node 1 socket 2 core 3 meters
         node 1 socket 2 core 4
         node 1 socket 2 core 4 meters offset array
         node 1 socket 2 core 4 meters
 node 2
 node 2 meters offset array
 node 2 socket offset array
 node 2 meters
    node 2 socket 1
    node 2 socket 1 meters offset array
    node 2 socket 1 core offset array
    node 2 socket 1 meters
         node 2 socket 1 core 1
         node 2 socket 1 core 1 meters offset array
         node 2 socket 1 core 1 meters
         node 2 socket 1 core 2
         node 2 socket 1 core 2 meters offset array
         node 2 socket 1 core 2 meters
         node 2 socket 1 core 3
         node 2 socket 1 core 3 meters offset array
         node 2 socket 1 core 3 meters
         node 2 socket 1 core 4
         node 2 socket 1 core 4 meters offset array
         node 2 socket 1 core 4 meters
    node 2 socket 2
    node 2 socket 2 meters offset array
    node 3 socket 2 core offset array
    node 2 socket 2 meters
         node 2 socket 2 core 1
         node 2 socket 2 core 1 meters offset array
         node 2 socket 2 core 1 meters
         node 2 socket 2 core 2
         node 2 socket 2 core 2 meters offset array
         node 2 socket 2 core 2 meters
         node 2 socket 2 core 3
         node 2 socket 2 core 3 meters offset array
         node 2 socket 2 core 3 meters
         node 2 socket 2 core 4
         node 2 socket 2 core 4 meters offset array
         node 2 socket 2 core 4 meters

*/

#define RCRFILE_NAME "RCRFile"

const int RCRBlackBoardVersionNumber = 1; // incremented each time any fields are changed
const int MAX_RCRFILE_SIZE = 8192; 
const int NODENAMEMAXLENGTH = 256;

struct _RCRBBMetaData
{
  int64_t bbVersionNumber;
  int64_t bbTreeSize;
};

struct  _RCRBlackboard
{
  struct _RCRBBMetaData md;
  int64_t               numBBMeters;
  int64_t               bbMeterOffset; // offset of array of meter offsets
  int64_t               numNodes;
  int64_t               nodeOffset;  // offset of array of node offsets
};

struct _RCRNodeMetaData
{
  int64_t nodeType;
  int64_t nodeNumber;
  int64_t nodeTreeSize;
  char *  nodeName; //Node unique id. TODO: Change to double pointer.

};

struct _RCRNode 
{
  struct _RCRNodeMetaData md;
  int64_t                 numNodeMeters;
  int64_t                 nodeMeterOffset; // offset of array of meter offsets
  int64_t                 numSockets;
  int64_t                 socketOffset;  // offset of array of socket offsets
};

struct _RCRSocketMetaData
{
  int64_t socketType;
  int64_t socketNumber;
  int64_t socketTreeSize;
};

struct _RCRSocket
{
  struct _RCRSocketMetaData md;
  int64_t                   numSocketMeters;
  int64_t                   socketMeterOffset; // offset of array of meter offsets
  int64_t                   numCores;
  int64_t                   coreOffset;  // offset of array of core offsets
};
      
struct _RCRCoreMetaData
{
  int64_t coreType;
  int64_t coreNumber;
  int64_t coreTreeSize;
};

struct _RCRCore   // leaf hardware element -- only meters no children
{
  struct _RCRCoreMetaData md;
  int64_t                 numCoreMeters;
  int64_t                 coreMeterOffset; // offset of array of meter offsets
  int64_t                 numThreads;
  int64_t                 threadOffset;  // offset of array of core offsets
};

struct _RCRThreadMetaData
{
  int64_t threadType;
  int64_t threadNumber;
  int64_t threadTreeSize;
};


struct _RCRThread   // leaf hardware element -- only meters no children
{
  struct _RCRThreadMetaData md;
  int64_t                 numThreadMeters;
  int64_t                 threadMeterOffset; // offset of array of meter offsets
};

struct _RCROffset
{
  int64_t  size;
  int64_t  offset[]; // id meter type
};

struct _RCRMeterValue
{
  int64_t meterID; // id meter type
                   // -- smaller field??  only if something else filling the 64 bits
  union{           // union for basically any 64-bit performance data to be saved
    volatile int64_t  iValue;
    volatile uint64_t uValue;
    volatile double   fValue;
  } data;
};

struct _RCRMeter
{
  int64_t size;
  struct _RCRMeterValue value[]; // array of meters and their values
};

typedef struct _RCRMeter RCRMeter;

// defines for meter types

enum rcrMeterType {DUMMY_SYSTEM,
		   DUMMY_NODE, 
		   DUMMY_SOCKET, 
		   DUMMY_CORE, 
		   DUMMY_THREAD,
		   MEMORY_CONCURRENCY,
		   TEMP_STATUS,
		   ENERGY_STATUS,
		   TOR_OCCUPANCY,
		   UNCORE_CLOCK_TICKS,
                   VICTIMIZED_LINES,
		   END};

// string names for meter types -- index must match  #define value above
/*
const char * typeString[] = { "memory concurrency", // 1
                          "end"}; // strings for printing
*/


// defines for Node Type

// defines for Socket Type

// defines for Core Type

// defines for Thread Type

#endif 
