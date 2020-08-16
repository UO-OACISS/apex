/*  Copyright (c) 2020 University of Oregon
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "apex_nvml.hpp"

#define NVML_CALL(call)                                                      \
do {                                                                         \
    nvmlResult _status = call;                                               \
    if (_status != NVML_SUCCESS) {                                           \
        const char *errstr;                                                  \
        nvmlGetResultString(_status, &errstr);                               \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                __FILE__, __LINE__, #call, errstr);                          \
        exit(-1);                                                            \
    }                                                                        \
} while (0)

namespace apex { namespace nvml {

monitor::monitor (void) {
    NVML_CALL(nvmlInit_v2());
    // get the unit count
    NVML_CALL(nvmlUnitGetCount(&unitCount));
    // get the total device count
    NVML_CALL(nvmlDeviceGetCount_v2 (&totalDeviceCount));
    // get the unit handles
    for (int i = 0 ; i < unitCount ; i++) {
        nvmlUnit_t unit;
        NVML_CALL(nvmlUnitGetHandleByIndex(i, &unit));
        units.push_back(unit);
        uint32_t unitDeviceCount;
        nvmlDevice_t unitDevices[totalDeviceCount];
        NVML_CALL(nvmlUnitGetDevices (unit, unitDeviceCount, unitDevices));
        // get the device handles for each unit
        for (int j = 0 ; j < unitDeviceCount ; j++) {
            devices.push_back(std::move<nvmlDevice_t>(unitDevices[j]));
        }
    }
}
monitor::~monitor (void) {
    NVML_CALL(nvmlShutdown());
}

void monitor::query(void) {
    nvmlUnit_t unit;
    uint32_t type;
    unsigned int* temp;
    for (size_t d = 0 ; d < devices.size() ; d++) {
        nvmlUtilization_t utilization;
        NVML_CALL(nvmlDeviceGetUtilizationRates(devices[d], &utilization);
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU utilization";
            std::string tmp{ss.str()};
            double value = (double)(utilization.gpu);
            apex::sample_value(tmp, value);
        }
        {
            std::stringstream ss;
            ss << "Device " << d << " Memory utilization";
            std::string tmp{ss.str()};
            double value = (double)(utilization.memory);
            apex::sample_value(tmp, value);
        }
    }
}

} // namespace nvml
} // namespace apex
