/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
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
        fprintf(stderr, "%s:%d: error: function %s failed with error %d: %s.\n", \
                __FILE__, __LINE__, #call, _status, errstr);                 \
        exit(-1);                                                            \
    }                                                                        \
} while (0);

#define MILLIONTH 1.0e-6 // scale to MB
#define BILLIONTH 1.0e-9 // scale to GB
#define PCIE_THROUGHPUT 1.0e-3  // to scale KB to MB
#define NVLINK_BW 1.0e-3  // to scale MB/s to GB/s
#define WATTS 1.0e-3  // scale mW to W

namespace apex { namespace nvml {

std::set<uint32_t> monitor::activeDeviceIndices;
std::mutex monitor::indexMutex;

monitor::monitor (void) {
    if (!apex_options::monitor_gpu()) {
        success = false;
        return;
    }
    success = true;
    NVML_CALL(nvmlInit_v2());
    // get the total device count
    NVML_CALL(nvmlDeviceGetCount_v2(&deviceCount));
    char version_name[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE];
    NVML_CALL(nvmlSystemGetNVMLVersion(version_name, NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE));
    DEBUG_PRINT("NVML Version %s Found %u total devices\n", version_name, deviceCount);

    devices.reserve(deviceCount);
    // get the unit handles
    for (uint32_t i = 0 ; i < deviceCount ; i++) {
        nvmlDevice_t device;
        NVML_CALL(nvmlDeviceGetHandleByIndex_v2(i, &device));
        devices.push_back(device);
        queried_once.push_back(false);
    }
    // assume the first device is used by default
    activateDeviceIndex(0);
}
monitor::~monitor (void) {
    if (success) {
        NVML_CALL(nvmlShutdown());
    }
}

void monitor::query(void) {
    if (!success) { return; }
    indexMutex.lock();
    // use the copy constructor to get the set of active indices
    std::set<uint32_t> indexSet{activeDeviceIndices};
    indexMutex.unlock();

    for (uint32_t d : indexSet) {
        /* Get overall utilization percentages */
        nvmlUtilization_t utilization;
        nvmlDevice_t dev = devices[d];
        NVML_CALL(nvmlDeviceGetUtilizationRates(dev, &utilization));
        {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Utilization %";
            std::string tmp{ss.str()};
            double value = (double)(utilization.gpu);
            sample_value(tmp, value);
        }
        {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Memory Utilization %";
            std::string tmp{ss.str()};
            double value = (double)(utilization.memory);
            sample_value(tmp, value);
        }

        /* Get memory bytes allocated */
        nvmlMemory_t memory;
        NVML_CALL(nvmlDeviceGetMemoryInfo(devices[d], &memory));
        if (!queried_once[d])
        {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Memory Total (GB)";
            std::string tmp{ss.str()};
            double value = (double)(memory.total) * BILLIONTH;
            sample_value(tmp, value);
        }
        {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Memory Free (GB)";
            std::string tmp{ss.str()};
            double value = (double)(memory.free) * BILLIONTH;
            sample_value(tmp, value);
        }
        {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Memory Used (GB)";
            std::string tmp{ss.str()};
            double value = (double)(memory.used) * BILLIONTH;
            sample_value(tmp, value);
        }

        /* Get clock settings */
        uint32_t clock = 0;
        NVML_CALL(nvmlDeviceGetClock(devices[d], NVML_CLOCK_SM,
            NVML_CLOCK_ID_CURRENT, &clock));
        {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Clock SM (MHz)";
            std::string tmp{ss.str()};
            double value = (double)(clock);
            sample_value(tmp, value);
        }
        NVML_CALL(nvmlDeviceGetClock(devices[d], NVML_CLOCK_MEM,
            NVML_CLOCK_ID_CURRENT, &clock));
        {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Clock Memory (MHz)";
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
            ss << "GPU: Device " << d << " Throttle Idle";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonHwPowerBrakeSlowdown) {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Throttle Power Break Slowdown";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonHwSlowdown) {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Throttle HW Break Slowdown";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonHwThermalSlowdown) {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Throttle HW Thermal Slowdown";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonSwPowerCap) {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Throttle SW Power Cap Slowdown";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonSwThermalSlowdown) {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Throttle SW Thermal Slowdown";
            std::string tmp{ss.str()};
            sample_value(tmp, 1);
        }
        if (reasons && nvmlClocksThrottleReasonSyncBoost) {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Throttle Sync Boost";
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
            ss << "GPU: Device " << d << " Fan Speed %";
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
            ss << "GPU: Device " << d << " PCIe TX Throughput (MB/s)";
            std::string tmp{ss.str()};
            double value = (double)(throughput) * PCIE_THROUGHPUT;
            sample_value(tmp, value);
        }
        NVML_CALL(nvmlDeviceGetPcieThroughput(devices[d],
            NVML_PCIE_UTIL_RX_BYTES, &throughput));
        {
            std::stringstream ss;
            ss << "GPU: Device " << d << " PCIe RX Throughput (MB/s)";
            std::string tmp{ss.str()};
            double value = (double)(throughput) * PCIE_THROUGHPUT;
            sample_value(tmp, value);
        }

        uint32_t power = 0;
        NVML_CALL(nvmlDeviceGetPowerUsage(devices[d], &power));
        {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Power (W)";
            std::string tmp{ss.str()};
            double value = (double)(power) * WATTS;
            sample_value(tmp, value);
        }

        uint32_t temperature = 0;
        NVML_CALL(nvmlDeviceGetTemperature(devices[d], NVML_TEMPERATURE_GPU,
            &temperature));
        {
            std::stringstream ss;
            ss << "GPU: Device " << d << " Temperature (C)";
            std::string tmp{ss.str()};
            double value = (double)(temperature);
            sample_value(tmp, value);
        }

        if (!queried_once[d]) {
            nvmlFieldValue_t values[2];
            int valuesCount{2};
            values[0].fieldId = NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON;
            values[1].fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;
            NVML_CALL(nvmlDeviceGetFieldValues(devices[d],
                valuesCount, values));
            if (values[0].nvmlReturn == NVML_SUCCESS)
                {
                    std::stringstream ss;
                    ss << "GPU: Device " << d << " NvLink Speed (GB/s)";
                    std::string tmp{ss.str()};
                    double value = convertValue(values[0]) * NVLINK_BW;
                    sample_value(tmp, value);
                }
            if (values[1].nvmlReturn == NVML_SUCCESS)
                {
                    std::stringstream ss;
                    ss << "GPU: Device " << d << " NvLink Link Count";
                    std::string tmp{ss.str()};
                    double value = convertValue(values[1]);
                    sample_value(tmp, value);
                }
        }
#if defined(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX)
        nvmlFieldValue_t values[6];
        values[0].fieldId = NVML_FI_DEV_NVLINK_BANDWIDTH_C0_TOTAL;
        values[1].fieldId = NVML_FI_DEV_NVLINK_BANDWIDTH_C1_TOTAL;
        values[2].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
        values[3].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
        values[4].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX;
        values[5].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX;
        values[2].scopeId = UINT_MAX;
        values[3].scopeId = UINT_MAX;
        values[4].scopeId = UINT_MAX;
        values[5].scopeId = UINT_MAX;
        int valuesCount{6};
#else
        nvmlFieldValue_t values[2];
        values[0].fieldId = NVML_FI_DEV_NVLINK_BANDWIDTH_C0_TOTAL;
        values[1].fieldId = NVML_FI_DEV_NVLINK_BANDWIDTH_C1_TOTAL;
        int valuesCount{2};
#endif
        NVML_CALL(nvmlDeviceGetFieldValues(devices[d],
            valuesCount, values));
        if (values[0].nvmlReturn == NVML_SUCCESS)
            {
                std::stringstream ss;
                ss << "GPU: Device " << d << " NvLink Bandwidth C0 Total";
                std::string tmp{ss.str()};
                double value = convertValue(values[0]);
                sample_value(tmp, value);
            }
        if (values[1].nvmlReturn == NVML_SUCCESS)
            {
                std::stringstream ss;
                ss << "GPU: Device " << d << " NvLink Bandwidth C1 Total";
                std::string tmp{ss.str()};
                double value = convertValue(values[1]);
                sample_value(tmp, value);
            }
#if defined(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX)
        if (values[2].nvmlReturn == NVML_SUCCESS)
            {
                std::stringstream ss;
                ss << "GPU: Device " << d << " NvLink Throughput Data TX";
                std::string tmp{ss.str()};
                double value = convertValue(values[2]);
                sample_value(tmp, value);
            }
        if (values[3].nvmlReturn == NVML_SUCCESS)
            {
                std::stringstream ss;
                ss << "GPU: Device " << d << " NvLink Throughput Data RX";
                std::string tmp{ss.str()};
                double value = convertValue(values[3]);
                sample_value(tmp, value);
            }
        if (values[4].nvmlReturn == NVML_SUCCESS)
            {
                std::stringstream ss;
                ss << "GPU: Device " << d << " NvLink Throughput Raw TX";
                std::string tmp{ss.str()};
                double value = convertValue(values[4]);
                sample_value(tmp, value);
            }
        if (values[5].nvmlReturn == NVML_SUCCESS)
            {
                std::stringstream ss;
                ss << "GPU: Device " << d << " NvLink Throughput Raw RX";
                std::string tmp{ss.str()};
                double value = convertValue(values[5]);
                sample_value(tmp, value);
            }
#endif
        queried_once[d] = true;
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
