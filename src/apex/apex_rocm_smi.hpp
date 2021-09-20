/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include "apex.hpp"
#include "rocm_smi/rocm_smi.h"
#include <set>
#include <mutex>
#include <sstream>
#include <string>

namespace apex { namespace rsmi {

class DeviceInfo {
public:
    static const size_t str_len = 256;
    uint16_t id;
    char sku[str_len];
    uint16_t vendor_id;
    char name[str_len];
    char brand[str_len];
    char vendor_name[str_len];
    char vram_vendor_name[str_len];
    char serial_number[str_len];
    uint16_t subsystem_id;
    char subsystem_name[str_len];
    uint32_t drm_minor_id;
    uint16_t subsystem_vendor_id;
    uint64_t unique_id;
    std::string to_string(void) {
        std::stringstream ss;
        ss << "Device: " "\n"
           << "  ID: " << id << "\n"
           << "  Unique ID: " << unique_id << "\n"
           << "  Sku: " << sku << "\n"
           << "  Name: " << name << "\n"
           << "  Brand: " << brand << "\n"
           << "  Vendor ID: " << vendor_id << "\n"
           << "  Vendor Name: " << vendor_name << "\n"
           << "  VRAM Vendor Name: " << vram_vendor_name << "\n"
           << "  Serial Number: " << serial_number << "\n"
           << "  Subsystem ID: " << subsystem_id << "\n"
           << "  Subsystem Name: " << subsystem_name << "\n"
           << "  Subsystem Vendor ID: " << subsystem_vendor_id << "\n"
           << "  DRM Minor  ID: " << drm_minor_id;
         std::string tmp{ss.str()};
         return tmp;
    }
};


class monitor {
public:
    monitor (void);
    ~monitor (void);
    void query();
    static void activateDeviceIndex(uint32_t index);
private:
    uint32_t deviceCount;
    std::vector<uint64_t> devices;
    std::vector<DeviceInfo> deviceInfos;
    std::vector<bool> queried_once;
    static std::set<uint32_t> activeDeviceIndices;
    static std::mutex indexMutex;
    //double convertValue(nvmlFieldValue_t &value);
}; // class monitor

} // namespace rsmi
} // namespace apex
