#include <cstdint>

typedef struct KokkosPDeviceInfo {
  uint32_t deviceID;
} KokkosPDeviceInfo_t;

/* This handle describes a Kokkos memory space. The name member is a
 * zero-padded string which currently can take the values "Host" or "Cuda".
 */
typedef struct SpaceHandle {
  char name[64];
} SpaceHandle_t;
