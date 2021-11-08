/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "hip_profiler.hpp"
#include "utils.hpp"
#include "apex_assert.h"
#include <string.h>
#include <unistd.h>
#include <iostream>
#include "ctrl/run_kernel.h"
#include "ctrl/test_aql.h"
#include "ctrl/test_hsa.h"
#include "util/hsa_rsrc_factory.h"
#include "util/test_assert.h"

#define ROCPROFILER_CALL(call)                                                      \
do {                                                                         \
    hsa_status_t _status = call;                                            \
    if (_status != HSA_STATUS_SUCCESS) {                                    \
        const char *errstr;                                                  \
        if (rsmi_status_string(_status, &errstr) == ROCPROFILER_STATUS_SUCCESS) {   \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d: %s.\n", \
                __FILE__, __LINE__, #call, _status, errstr);                 \
        exit(-1);                                                            \
        }                                                                    \
    }                                                                        \
} while (0);

#define ROCPROFILER_CALL_NOEXIT(call)                                               \
do {                                                                         \
    hsa_status_t _status = call;                                            \
    if (_status != HSA_STATUS_SUCCESS) {                                    \
        const char *errstr;                                                  \
        if (rocprofiler_error_string(&errstr) == ROCPROFILER_STATUS_SUCCESS) {   \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d: %s.\n", \
                __FILE__, __LINE__, #call, _status, errstr);                 \
        success = false;                                                     \
        }                                                                    \
    }                                                                        \
} while (0);

namespace apex { namespace rocprofiler {

monitor::monitor (void) : enabled(false), status(HSA_STATUS_SUCCESS), context(nullptr) {
    // if disabled, do nothing...
    if (!apex_options::use_hip_profiler()) {
        return;
    }
    memset(feature, 0, sizeof(feature));
    std::stringstream metric_ss(apex_options::rocprof_metrics());
    feature_count = 0;
    while(metric_ss.good()) {
        std::string metric;
        getline(metric_ss, metric, ','); // tokenize by comma
        feature[feature_count].kind = ROCPROFILER_FEATURE_KIND_METRIC;
        feature[feature_count].name = strdup(metric.c_str());
        feature_count++;
    }

  // Instantiate HSA resources
  HsaRsrcFactory::Create();

  // Getting GPU device info
  const AgentInfo* agent_info;
  //if (HsaRsrcFactory::Instance().GetGpuAgentInfo(0, &agent_info) == false) abort();
  agent_info = HsaRsrcFactory::Instance().GetGpuAgentInfo(0);
  if (agent_info == nullptr) { abort(); };

  // Creating profiling context - we want to monitor ALL queues with this...
  properties = {};
  properties.queue_depth = 128;
  status = rocprofiler_open(agent_info->dev_id, feature, feature_count, &context,
                            ROCPROFILER_MODE_STANDALONE |
                            ROCPROFILER_MODE_CREATEQUEUE |
                            ROCPROFILER_MODE_SINGLEGROUP, &properties);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);

  // Test initialization
  TestHsa::HsaInstantiate();

  // Dispatching profiled kernel n-times to collect all counter groups data
  group_n = 0;
  status = rocprofiler_start(context, group_n);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);
  //std::cout << "start" << std::endl;
  enabled = true;
}

void monitor::stop (void) {
    enabled = false;
    // if disabled, do nothing...
    if (!apex_options::use_hip_profiler()) {
        return;
    }
  // Stop counters
  std::cout << "stop..." << std::endl;
  status = rocprofiler_stop(context, group_n);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);
}

monitor::~monitor (void) {
    enabled = false;
    // if disabled, do nothing...
    if (!apex_options::use_hip_profiler()) {
        return;
    }
  // Finishing cleanup
  // Deleting profiling context will delete all allocated resources
  std::cout << "close..." << std::endl;
  //status = rocprofiler_close(context);
  //TEST_STATUS(status == HSA_STATUS_SUCCESS);
  std::cout << "done." << std::endl;
}

// print profiler features
void print_features(rocprofiler_feature_t* feature, uint32_t feature_count) {
    for (rocprofiler_feature_t* p = feature; p < feature + feature_count; ++p) {
      //std::cout << (p - feature) << ": " << p->name;
      std::stringstream ss;
      ss << "GPU METRIC: " << p->name;
      switch (p->data.kind) {
        case ROCPROFILER_DATA_KIND_INT64:
          //std::cout << std::dec << " result64 (" << p->data.result_int64 << ")" << std::endl;
          sample_value(ss.str(), (double)(p->data.result_int64));
          break;
        case ROCPROFILER_DATA_KIND_BYTES: {
          const char* ptr = reinterpret_cast<const char*>(p->data.result_bytes.ptr);
          uint64_t size = 0;
          for (unsigned i = 0; i < p->data.result_bytes.instance_count; ++i) {
            size = *reinterpret_cast<const uint64_t*>(ptr);
            const char* data = ptr + sizeof(size);
            //std::cout << std::endl;
            //std::cout << std::hex << "  data (" << (void*)data << ")" << std::endl;
            //std::cout << std::dec << "  size (" << size << ")" << std::endl;
            ptr = data + size;
            ss << " Bytes ";
            sample_value(ss.str(), (double)(size));
          }
          break;
        }
        default:
          std::cout << "result kind (" << p->data.kind << ")" << std::endl;
          TEST_ASSERT(false);
      }
    }
}


void monitor::query(void) {
    // if disabled, do nothing...
    if (!apex_options::use_hip_profiler() || !enabled) {
        return;
    }
    //std::cout << "read features" << std::endl;
    hsa_status_t status = rocprofiler_read(context, group_n);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);
    //std::cout << "read issue..." << std::endl;
    status = rocprofiler_get_data(context, group_n);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);
    status = rocprofiler_get_metrics(context);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);
    print_features(feature, feature_count);
    //std::cout << "done" << std::endl;
}

} // namespace rocprofiler
} // namespace apex
