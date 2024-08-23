/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the BSD 2-Clause Software License. (See accompanying
 * file LICENSE)
 */

#define _GNU_SOURCE
#include <sys/syscall.h>
#include <unistd.h>
#include <stddef.h>
#include <atomic>
#include <array>
#include <algorithm>
#include <iterator>
#include <string.h>
#include "timer_plugin/tasktimer.h"

/* ISO C doesn't allow __PRETTY_FUNCTION__, so only do it with C++ */
#if defined(__GNUC__) && defined(__cplusplus)
#define __APEX__FUNCTION__ __PRETTY_FUNCTION__
#else
#define __APEX__FUNCTION__ __func__
#endif

uint64_t _my_gettid(void) {
    pid_t x = syscall(SYS_gettid);
    return (uint64_t)(x);
}

/* This simple example is truly overkill, but it tests all aspects of the API. */

std::atomic<uint64_t> guid{0};

void A(uint64_t);
void B(uint64_t, uint64_t);
void C(uint64_t, uint64_t);
void D(void);
void E(void);
void F(void);
void xfer(void);

void A(uint64_t parent) {
    uint64_t parents[] = {parent};
    uint64_t myguid = guid++;
    // both address and name
    TASKTIMER_CREATE(&A, __APEX__FUNCTION__, myguid, parents, 1, tt_A);
    tasktimer_argument_value_t args[1];
    args[0].type = TASKTIMER_LONG_INTEGER_TYPE;
    args[0].l_value = parent;
    TASKTIMER_SCHEDULE(tt_A, args, 1);
    tasktimer_execution_space_t resource;
    resource.type = TASKTIMER_DEVICE_CPU;
    resource.device_id = 0;
    resource.instance_id = _my_gettid();
    TASKTIMER_START(tt_A, &resource);
    B(parent, myguid);
    C(parent, myguid);
    TASKTIMER_STOP(tt_A);
}

void B(uint64_t parent1, uint64_t parent2) {
    uint64_t parents[] = {parent1, parent2};
    uint64_t myguid = guid++;
    // both address and name
    TASKTIMER_CREATE(&B, __APEX__FUNCTION__, myguid, parents, 2, tt_B);
    tasktimer_argument_value_t args[2];
    args[0].type = TASKTIMER_LONG_INTEGER_TYPE;
    args[0].l_value = parent1;
    args[1].type = TASKTIMER_LONG_INTEGER_TYPE;
    args[1].l_value = parent2;
    TASKTIMER_SCHEDULE(tt_B, args, 2);
    tasktimer_execution_space_t resource;
    resource.type = TASKTIMER_DEVICE_CPU;
    resource.device_id = 0;
    resource.instance_id = _my_gettid();
    TASKTIMER_START(tt_B, &resource);
    TASKTIMER_STOP(tt_B);
}

void C(uint64_t parent1, uint64_t parent2) {
    uint64_t parents[] = {parent1, parent2};
    uint64_t myguid = guid++;
    // no name, just address
    TASKTIMER_CREATE(&C, nullptr, myguid, parents, 2, tt_C);
    tasktimer_argument_value_t args[2];
    args[0].type = TASKTIMER_LONG_INTEGER_TYPE;
    args[0].l_value = parent1;
    args[1].type = TASKTIMER_LONG_INTEGER_TYPE;
    args[1].l_value = parent2;
    TASKTIMER_SCHEDULE(tt_C, args, 2);
    tasktimer_execution_space_t resource;
    resource.type = TASKTIMER_DEVICE_CPU;
    resource.device_id = 0;
    resource.instance_id = _my_gettid();
    TASKTIMER_START(tt_C, &resource);
    D();
    xfer();
    E();
    xfer();
    F();
    TASKTIMER_STOP(tt_C);
}

void D(void) {
    TASKTIMER_COMMAND_START(__APEX__FUNCTION__);
    TASKTIMER_COMMAND_STOP();
}

void E(void) {
    TASKTIMER_COMMAND_START(__APEX__FUNCTION__);
    TASKTIMER_COMMAND_STOP();
}

void F(void) {
    TASKTIMER_COMMAND_START(__APEX__FUNCTION__);
    TASKTIMER_COMMAND_STOP();
}

void xfer(void) {
    constexpr uint64_t maxlen = 1024;
    std::array<uint64_t, maxlen> source{1};
    std::array<uint64_t, maxlen> dest{0};
    tasktimer_execution_space_t source_info, dest_info;
    tasktimer_execution_space_p sip = &source_info;
    tasktimer_execution_space_p dip = &dest_info;
    source_info.type = TASKTIMER_DEVICE_CPU;
    source_info.device_id = 0;
    source_info.instance_id = 0;
    dest_info.type = TASKTIMER_DEVICE_CPU;
    dest_info.device_id = 0;
    dest_info.instance_id = 0;
    TASKTIMER_DATA_TRANSFER_START(100, sip, "source", source.data(), dip, "dest", dest.data());
    std::copy(std::begin(source), std::end(source), std::begin(dest));
    TASKTIMER_DATA_TRANSFER_STOP(100);
}

tasktimer_execution_space_t make_resource(void){
    tasktimer_execution_space_t resource;
    resource.type = TASKTIMER_DEVICE_CPU;
    resource.device_id = 0;
    resource.instance_id = _my_gettid();
    return resource;
}

void add_parent_test(uint64_t parent) {
    uint64_t parents[] = {parent};
    uint64_t myguid = guid++;
    // both address and name
    TASKTIMER_CREATE(nullptr, __APEX__FUNCTION__, myguid, parents, 1, tt_add_parent_test);
    TASKTIMER_SCHEDULE(tt_add_parent_test, nullptr, 0);
    auto resource = make_resource();
    TASKTIMER_START(tt_add_parent_test, &resource);
    // make a new timer with no parent
    uint64_t newparent = guid++;
    TASKTIMER_CREATE(nullptr, "added_parent", newparent, nullptr, 0, tt_newparent);
    TASKTIMER_ADD_PARENTS(tt_newparent, parents, 1);
    TASKTIMER_SCHEDULE(tt_newparent, nullptr, 0);
    TASKTIMER_START(tt_newparent, &resource);
    TASKTIMER_STOP(tt_newparent);
    TASKTIMER_STOP(tt_add_parent_test);
}

void add_child_test(tasktimer_timer_t parent) {
    // create without a parent
    uint64_t myguid = guid++;
    TASKTIMER_CREATE(nullptr, __APEX__FUNCTION__, myguid, nullptr, 0, tt_add_child_test);
    TASKTIMER_SCHEDULE(tt_add_child_test, nullptr, 0);
    auto resource = make_resource();
    TASKTIMER_START(tt_add_child_test, &resource);
    // make another timer with no parent
    uint64_t newchild = guid++;
    TASKTIMER_CREATE(nullptr, "added_child", newchild, nullptr, 0, tt_newchild);
    uint64_t children[] = {myguid,newchild};
    TASKTIMER_ADD_CHILDREN(parent, children, 2);
    TASKTIMER_SCHEDULE(tt_newchild, nullptr, 0);
    TASKTIMER_START(tt_newchild, &resource);
    TASKTIMER_STOP(tt_newchild);
    TASKTIMER_STOP(tt_add_child_test);
}

int main(int argc, char * argv[]) {
    // initialize the timer plugin
    TASKTIMER_INITIALIZE();
    uint64_t myguid = guid++;
    // no address, just name
    TASKTIMER_CREATE(nullptr, __APEX__FUNCTION__, myguid, nullptr, 0, tt);
    // schedule the task
    TASKTIMER_SCHEDULE(tt, nullptr, 0);
    // execute the task on CPU 0, thread_id
    tasktimer_execution_space_t resource;
    resource.type = TASKTIMER_DEVICE_CPU;
    resource.device_id = 0;
    resource.instance_id = _my_gettid();
    TASKTIMER_START(tt, &resource);
    // yield the task
    TASKTIMER_YIELD(tt);
    // run a "child" task
    A(myguid);
    // test the "add_parent" feature
    add_parent_test(myguid);
    // test the "add_child" feature
    add_child_test(tt);
    // resume the task
    TASKTIMER_RESUME(tt, &resource);
    // stop the task
    TASKTIMER_STOP(tt);
    // finalize the timer plugin
    TASKTIMER_FINALIZE();
    return 0;
}
