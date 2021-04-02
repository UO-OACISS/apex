/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>

#if !defined(__APPLE__)
#define APEX_PTHREAD_BARRIER_AVAILABLE
#endif

typedef void * (*start_routine_p)(void *);
typedef int (*pthread_create_p)(pthread_t *, const pthread_attr_t *, start_routine_p, void *arg);
typedef int (*pthread_join_p)(pthread_t, void **);
#if 0
typedef void (*pthread_exit_p)(void *);
#if defined(APEX_PTHREAD_BARRIER_AVAILABLE)
typedef int (*pthread_barrier_wait_p)(pthread_barrier_t *);
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

int apex_pthread_create_wrapper(pthread_create_p pthread_create_call,
    pthread_t * threadp, const pthread_attr_t * attr, start_routine_p, void * arg);
int apex_pthread_join_wrapper(pthread_join_p pthread_join_call, pthread_t thread, void **retval);
#if 0
void apex_pthread_exit_wrapper(pthread_exit_p pthread_exit_call, void * value_ptr);
#if defined(APEX_PTHREAD_BARRIER_AVAILABLE)
int apex_pthread_barrier_wait_wrapper(pthread_barrier_wait_p pthread_barrier_wait_call,
    pthread_barrier_t * barrier);
#endif
#endif

#ifdef __cplusplus
}
#endif

#define APEX_PRELOAD_LIB
