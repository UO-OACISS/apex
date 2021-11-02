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
    feature_count = 9;
    memset(feature, 0, sizeof(feature));
    feature[0].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[0].name = "VALUUtilization";
    feature[1].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[1].name = "VALUBusy";
    feature[2].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[2].name = "SALUBusy";
    feature[3].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[3].name = "L2CacheHit";
    feature[4].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[4].name = "MemUnitBusy";
    feature[5].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[5].name = "MemUnitStalled";
    feature[6].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[6].name = "WriteUnitStalled";
    feature[7].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[7].name = "ALUStalledByLDS";
    feature[8].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[8].name = "LDSBankConflict";

  // Instantiate HSA resources
  HsaRsrcFactory::Create();

  // Getting GPU device info
  const AgentInfo* agent_info = NULL;
  if (HsaRsrcFactory::Instance().GetGpuAgentInfo(0, &agent_info) == false) abort();

  // Creating the queues pool
  const unsigned queue_count = 16;
  hsa_queue_t* queue[queue_count];
  for (unsigned queue_ind = 0; queue_ind < queue_count; ++queue_ind) {
    if (HsaRsrcFactory::Instance().CreateQueue(agent_info, 128, &queue[queue_ind]) == false) abort();
  }
  hsa_queue_t* prof_queue = queue[0];

  // Creating profiling context
  properties = {};
  properties.queue = prof_queue;
  status = rocprofiler_open(agent_info->dev_id, feature, feature_count, &context,
                            ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_SINGLEGROUP, &properties);
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

monitor::~monitor (void) {
    enabled = false;
    // if disabled, do nothing...
    if (!apex_options::use_hip_profiler()) {
        return;
    }
  // Stop counters
  status = rocprofiler_stop(context, group_n);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);
  //std::cout << "stop" << std::endl;

  // Finishing cleanup
  // Deleting profiling context will delete all allocated resources
  status = rocprofiler_close(context);
  TEST_STATUS(status == HSA_STATUS_SUCCESS);

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
    //std::cout << "read issue" << std::endl;
    status = rocprofiler_get_data(context, group_n);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);
    status = rocprofiler_get_metrics(context);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);
    print_features(feature, feature_count);
}

} // namespace rocprofiler
} // namespace apex
