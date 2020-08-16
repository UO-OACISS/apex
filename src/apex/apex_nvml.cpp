/*  Copyright (c) 2020 University of Oregon
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "apex_nvml.hpp"
#include "nvml.h"
#include "utils.hpp"

#define NVML_CALL(call)                                                      \
do {                                                                         \
    nvmlReturn_t _status = call;                                             \
    if (_status != NVML_SUCCESS) {                                           \
        const char *errstr = nvmlErrorString(_status);                       \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                __FILE__, __LINE__, #call, errstr);                          \
        exit(-1);                                                            \
    }                                                                        \
} while (0);

namespace apex { namespace nvml {

monitor::monitor (void) {
    NVML_CALL(nvmlInit_v2());
    // get the total device count
    NVML_CALL(nvmlDeviceGetCount_v2(&deviceCount));
    DEBUG_PRINT("Found %u total devices\n", deviceCount);
    devices.reserve(deviceCount);
    // get the unit handles
    for (uint32_t i = 0 ; i < deviceCount ; i++) {
        nvmlDevice_t device;
        NVML_CALL(nvmlDeviceGetHandleByIndex(i, &device));
        devices.push_back(device);
    }
}
monitor::~monitor (void) {
    NVML_CALL(nvmlShutdown());
}

void monitor::query(void) {
    for (size_t d = 0 ; d < devices.size() ; d++) {
        nvmlUtilization_t utilization;
        NVML_CALL(nvmlDeviceGetUtilizationRates(devices[d], &utilization));
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU utilization";
            std::string tmp{ss.str()};
            double value = (double)(utilization.gpu);
            sample_value(tmp, value);
        }
        {
            std::stringstream ss;
            ss << "Device " << d << " Memory utilization";
            std::string tmp{ss.str()};
            double value = (double)(utilization.memory);
            sample_value(tmp, value);
        }
    }
}

} // namespace nvml
} // namespace apex
