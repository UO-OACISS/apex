/*
 * Copyright (c) 2014-2020 Kevin Huck
 * Copyright (c) 2014-2020 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "apex_nvml.hpp"
#include "nvml.h"
#include "utils.hpp"
#include "apex_assert.h"

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

#define MILLIONTH 1.0e-6 // scale to MB
#define PCIE_THROUGHPUT 1.0e-3  // to scale KB to MB
#define WATTS 1.0e-3  // scale mW to W

namespace apex { namespace nvml {

std::set<uint32_t> monitor::activeDeviceIndices;
std::mutex monitor::indexMutex;

monitor::monitor (void) {
    NVML_CALL(nvmlInit_v2());
    // get the total device count
    NVML_CALL(nvmlDeviceGetCount_v2(&deviceCount));
    DEBUG_PRINT("Found %u total devices\n", deviceCount);
    devices.reserve(deviceCount);
    // get the unit handles
    for (uint32_t i = 0 ; i < deviceCount ; i++) {
        nvmlDevice_t device;
        NVML_CALL(nvmlDeviceGetHandleByIndex_v2(i, &device));
        devices.push_back(device);
    }
    // assume the first device is used by default
    activateDeviceIndex(0);
}
monitor::~monitor (void) {
    NVML_CALL(nvmlShutdown());
}

void monitor::query(void) {
    indexMutex.lock();
    // use the copy constructor to get the set of active indices
    auto indexSet{activeDeviceIndices};
    indexMutex.unlock();

    for (auto d : indexSet) {
        /* Get overall utilization percentages */
        nvmlUtilization_t utilization;
        NVML_CALL(nvmlDeviceGetUtilizationRates(devices[d], &utilization));
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU Utilization %";
            std::string tmp{ss.str()};
            double value = (double)(utilization.gpu);
            sample_value(tmp, value);
        }
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU Memory Utilization %";
            std::string tmp{ss.str()};
            double value = (double)(utilization.memory);
            sample_value(tmp, value);
        }

        /* Get memory bytes allocated */
        nvmlMemory_t memory;
        NVML_CALL(nvmlDeviceGetMemoryInfo(devices[d], &memory));
        /* Doesn't change, no need to capture this
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU Memory Total (MB)";
            std::string tmp{ss.str()};
            double value = (double)(memory.total) * MILLIONTH;
            sample_value(tmp, value);
        }
        */
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU Memory Free (MB)";
            std::string tmp{ss.str()};
            double value = (double)(memory.free) * MILLIONTH;
            sample_value(tmp, value);
        }
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU Memory Used (MB)";
            std::string tmp{ss.str()};
            double value = (double)(memory.used) * MILLIONTH;
            sample_value(tmp, value);
        }

        /* Get clock settings */
        uint32_t clock = 0;
        NVML_CALL(nvmlDeviceGetClock(devices[d], NVML_CLOCK_SM,
            NVML_CLOCK_ID_CURRENT, &clock));
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU Clock SM (MHz)";
            std::string tmp{ss.str()};
            double value = (double)(clock);
            sample_value(tmp, value);
        }
        NVML_CALL(nvmlDeviceGetClock(devices[d], NVML_CLOCK_MEM,
            NVML_CLOCK_ID_CURRENT, &clock));
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU Clock Memory (MHz)";
            std::string tmp{ss.str()};
            double value = (double)(clock);
            sample_value(tmp, value);
        }

        /* Get clock throttle reasons */
#if 0
        unsigned long long reasons = 0ULL;
        NVML_CALL(nvmlDeviceGetCurrentClocksThrottleReasons(devices[d],
            &reasons));
        if (reasons && nvmlClocksThrottleReasonGpuIdle) {
            std::stringstream ss;
            ss << "Device " << d << " GPU Throttle Idle";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonHwPowerBrakeSlowdown) {
            std::stringstream ss;
            ss << "Device " << d << " GPU Throttle Power Break Slowdown";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonHwSlowdown) {
            std::stringstream ss;
            ss << "Device " << d << " GPU Throttle HW Break Slowdown";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonHwThermalSlowdown) {
            std::stringstream ss;
            ss << "Device " << d << " GPU Throttle HW Thermal Slowdown";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonSwPowerCap) {
            std::stringstream ss;
            ss << "Device " << d << " GPU Throttle SW Power Cap Slowdown";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonSwThermalSlowdown) {
            std::stringstream ss;
            ss << "Device " << d << " GPU Throttle SW Thermal Slowdown";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonSyncBoost) {
            std::stringstream ss;
            ss << "Device " << d << " GPU Throttle Sync Boost";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
#endif

        /* Get fan speed? */
#if 0
        uint32_t speed;
        NVML_CALL(nvmlDeviceGetFanSpeed_v2(devices[d], 0, &speed));
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU Fan Speed %";
            std::string tmp{ss.str()};
            double value = (double)(clock);
            sample_value(tmp, value);
        }
#endif

        /* Get PCIe throughput */
        uint32_t throughput = 0;
        NVML_CALL(nvmlDeviceGetPcieThroughput(devices[d],
            NVML_PCIE_UTIL_TX_BYTES, &throughput));
        {
            std::stringstream ss;
            ss << "Device " << d << " PCIe TX Throughput (MB/s)";
            std::string tmp{ss.str()};
            double value = (double)(throughput) * PCIE_THROUGHPUT;
            sample_value(tmp, value);
        }
        NVML_CALL(nvmlDeviceGetPcieThroughput(devices[d],
            NVML_PCIE_UTIL_RX_BYTES, &throughput));
        {
            std::stringstream ss;
            ss << "Device " << d << " PCIe RX Throughput (MB/s)";
            std::string tmp{ss.str()};
            double value = (double)(throughput) * PCIE_THROUGHPUT;
            sample_value(tmp, value);
        }

        uint32_t power = 0;
        NVML_CALL(nvmlDeviceGetPowerUsage(devices[d], &power));
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU Power (W)";
            std::string tmp{ss.str()};
            double value = (double)(power) * WATTS;
            sample_value(tmp, value);
        }

        uint32_t temperature = 0;
        NVML_CALL(nvmlDeviceGetTemperature(devices[d], NVML_TEMPERATURE_GPU,
            &temperature));
        {
            std::stringstream ss;
            ss << "Device " << d << " GPU Temperature (C)";
            std::string tmp{ss.str()};
            double value = (double)(temperature);
            sample_value(tmp, value);
        }

        int valuesCount{4};
        nvmlFieldValue_t values[4];
        values[0].fieldId = NVML_FI_DEV_NVLINK_BANDWIDTH_C0_TOTAL;
        values[1].fieldId = NVML_FI_DEV_NVLINK_BANDWIDTH_C1_TOTAL;
        values[2].fieldId = NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON;
        values[3].fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;
        NVML_CALL(nvmlDeviceGetFieldValues(devices[d],
            valuesCount, values));
        if (convertValue(values[3]) > 0.0) {
            {
                std::stringstream ss;
                ss << "Device " << d << " GPU NvLink Utilization C0";
                std::string tmp{ss.str()};
                double value = convertValue(values[0]);
                sample_value(tmp, value);
            }
            {
                std::stringstream ss;
                ss << "Device " << d << " GPU NvLink Utilization C1";
                std::string tmp{ss.str()};
                double value = convertValue(values[1]);
                sample_value(tmp, value);
            }
            {
                std::stringstream ss;
                ss << "Device " << d << " GPU NvLink Speed MB/s";
                std::string tmp{ss.str()};
                double value = convertValue(values[2]);
                sample_value(tmp, value);
            }
            {
                std::stringstream ss;
                ss << "Device " << d << " GPU NvLink Link Count";
                std::string tmp{ss.str()};
                double value = convertValue(values[3]);
                sample_value(tmp, value);
            }
        }
    }
}

double monitor::convertValue(nvmlFieldValue_t &value) {
    if (value.valueType == NVML_VALUE_TYPE_DOUBLE) {
        return value.value.dVal;
    } else if (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT) {
        return (double)(value.value.uiVal);
    } else if (value.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG) {
        return (double)(value.value.ulVal);
    } else if (value.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG) {
        return (double)(value.value.ullVal);
    } else if (value.valueType == NVML_VALUE_TYPE_SIGNED_LONG_LONG) {
        return (double)(value.value.sllVal);
    }
    return 0.0;
}

void monitor::activateDeviceIndex(uint32_t index) {
    indexMutex.lock();
    activeDeviceIndices.insert(index);
    indexMutex.unlock();
}

} // namespace nvml
} // namespace apex
