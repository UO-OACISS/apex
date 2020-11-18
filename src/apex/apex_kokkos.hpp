/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

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
