//  Copyright (c) 2014-2018 University of Oregon
//  Copyright (c) 2014-2018 Kevin Huck
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ITTNOTIFY_H
#define ITTNOTIFY_H 1

#include <string>
#include <cstddef>
#include "ittnotify_export.h"

/*
 * For details, see
 * http://software.intel.com/sites/products/documentation/gpa/13.1/Overview.htm
 *
 */

#ifndef UINT64
#define UINT64 uint64_t
#endif

#define __itt_attr_mutex   2

enum __itt_metadata_type {
  __itt_metadata_u64,
  __itt_metadata_s64,
  __itt_metadata_u32,
  __itt_metadata_s32,
  __itt_metadata_u16,
  __itt_metadata_s16,
  __itt_metadata_float,
  __itt_metadata_double
};

enum __itt_relation {
  __itt_relation_is_dependent_on,
  __itt_relation_is_sibling_of,
  __itt_relation_is_parent_of,
  __itt_relation_is_continuation_of,
  __itt_relation_is_child_of,
  __itt_relation_is_continued_by,
  __itt_relation_is_predecessor_to
};

enum __itt_scope {
  __itt_scope_global,
  __itt_scope_track_group,
  __itt_scope_track,
  __itt_scope_task,
  __itt_scope_marker
};

enum __itt_track_group_type { __itt_track_group_type_normal = 0 };

enum __itt_track_type {__itt_track_type_normal = 0 };

struct ___itt_caller {
  std::string *name;
};

typedef struct ___itt_caller* __itt_caller;

typedef struct ___itt_string_handle
{
    const char* strA; /*!< Copy of original string in ASCII. */
#if defined(UNICODE) || defined(_UNICODE)
    const wchar_t* strW; /*!< Copy of original string in UNICODE. */
#else  /* UNICODE || _UNICODE */
    void* strW;
#endif /* UNICODE || _UNICODE */
    int   extra1; /*!< Reserved. Must be zero   */
    void* extra2; /*!< Reserved. Must be zero   */
    struct ___itt_string_handle* next;
} __itt_string_handle;

typedef struct ___itt_domain
{
    volatile int flags; /*!< Zero if disabled, non-zero if enabled. The meaning
    of different non-zero values is reserved to the runtime */
    const char* nameA;  /*!< Copy of original name in ASCII. */
#if defined(UNICODE) || defined(_UNICODE)
    const wchar_t* nameW; /*!< Copy of original name in UNICODE. */
#else  /* UNICODE || _UNICODE */
    void* nameW;
#endif /* UNICODE || _UNICODE */
    int   extra1; /*!< Reserved to the runtime */
    void* extra2; /*!< Reserved to the runtime */
    struct ___itt_domain* next;
} __itt_domain;

typedef struct ___itt_id
{
    unsigned long long d1, d2, d3;
} __itt_id;

static const __itt_id __itt_null = { 0, 0, 0 };

typedef struct ___itt_clock_domain {
  std::string *name;
} __itt_clock_domain;

typedef enum {
    __itt_marker_scope_unknown = 0,
    __itt_marker_scope_global,
    __itt_marker_scope_process,
    __itt_marker_scope_thread,
    __itt_marker_scope_task
} __itt_marker_scope;

typedef struct ___itt_track {
  std::string *name;
} __itt_track;

typedef struct ___itt_track_group {
  std::string *name;
} __itt_track_group;

typedef int __itt_mark_type;

typedef void* __itt_heap_function;

#ifdef __cplusplus
extern "C" {
#endif
  ITTNOTIFY_EXPORT __itt_domain* __itt_domain_create(char const*);
  ITTNOTIFY_EXPORT void __itt_frame_begin_v3(__itt_domain const* frame,
  __itt_id* id);
  ITTNOTIFY_EXPORT void __itt_frame_end_v3(__itt_domain const* frame, __itt_id*
  id);
  ITTNOTIFY_EXPORT void __itt_id_create(__itt_domain const*, __itt_id id);
  ITTNOTIFY_EXPORT void __itt_id_create_ex(__itt_domain const*,
      __itt_clock_domain* clock_domain, UINT64 timestamp, __itt_id* id);
  ITTNOTIFY_EXPORT void __itt_id_destroy(const __itt_domain * domain, __itt_id*
  id);
  ITTNOTIFY_EXPORT void __itt_id_destroy_ex(__itt_domain* domain,
      __itt_clock_domain* clock_domain, UINT64 timestamp, __itt_id* id);
  ITTNOTIFY_EXPORT __itt_id __itt_id_make(void*, unsigned long);
  ITTNOTIFY_EXPORT void __itt_metadata_add (const __itt_domain * domain,
      __itt_id id, __itt_string_handle * name, __itt_marker_scope scope);
  ITTNOTIFY_EXPORT void _itt_metadata_add_with_scope(const __itt_domain * domain,
      __itt_scope scope, __itt_string_handle *key, __itt_metadata_type type,
      size_t count, void *data);
  ITTNOTIFY_EXPORT void __itt_metadata_str_add (const __itt_domain * domain,
      __itt_id id, __itt_string_handle * key, const char * data, size_t length);
  ITTNOTIFY_EXPORT void _itt_metadata_str_add_with_scope(const __itt_domain *
      domain, __itt_scope scope, __itt_string_handle *key, const char * data,
      size_t length);
  ITTNOTIFY_EXPORT void __itt_relation_add (const __itt_domain * domain,
      __itt_id head, __itt_relation relation, __itt_id tail);
  ITTNOTIFY_EXPORT void __itt_relation_add_to_current (const __itt_domain * domain,
      __itt_relation relation, __itt_id tail);
  ITTNOTIFY_EXPORT void __itt__set_track(__itt_track * track);
  ITTNOTIFY_EXPORT __itt_string_handle* __itt_string_handle_create (const char
      * name);
  ITTNOTIFY_EXPORT void __itt_task_begin(__itt_domain const*, __itt_id,
      __itt_id, __itt_string_handle*);
  ITTNOTIFY_EXPORT void __itt_task_begin_ex(__itt_domain* domain,
      __itt_clock_domain* clock_domain, UINT64 timestamp, __itt_id id, __itt_id
      parentid, __itt_string_handle *name);
  ITTNOTIFY_EXPORT void __itt_task_begin_fn(const __itt_domain * domain,
      __itt_id taskid, __itt_id parentid, void * fn);
  ITTNOTIFY_EXPORT void __itt_task_end(__itt_domain const*);
  ITTNOTIFY_EXPORT void __itt_task_end_ex(__itt_domain const*);
  ITTNOTIFY_EXPORT void __itt_task_group(const __itt_domain * domain, __itt_id
      id, __itt_id parentid, __itt_string_handle * name);
  ITTNOTIFY_EXPORT void __itt_thread_set_name (const char * name);
  ITTNOTIFY_EXPORT __itt_track_group*
      __itt_track_group_create(__itt_string_handle* name,
      __itt_track_group_type type);
  ITTNOTIFY_EXPORT __itt_track* __itt_track_create(__itt_track_group* track_group,
      __itt_string_handle* name, __itt_track_type track_type);

// those not mentiond in the documentation

ITTNOTIFY_EXPORT void __itt_sync_create(void*, const char*, const char*, int);
ITTNOTIFY_EXPORT void __itt_sync_rename(void*, const char*);
ITTNOTIFY_EXPORT void __itt_sync_prepare(void*);
ITTNOTIFY_EXPORT void __itt_sync_acquired(void*);
ITTNOTIFY_EXPORT void __itt_sync_cancel(void*);
ITTNOTIFY_EXPORT void __itt_sync_releasing(void*);
ITTNOTIFY_EXPORT void __itt_sync_destroy(void*);
ITTNOTIFY_EXPORT void __itt_thread_ignore();
ITTNOTIFY_EXPORT __itt_heap_function __itt_heap_function_create(const char*,
const char*);
ITTNOTIFY_EXPORT void __itt_heap_allocate_begin(__itt_heap_function,
std::size_t, int);
ITTNOTIFY_EXPORT void __itt_heap_allocate_end(__itt_heap_function, void**,
std::size_t, int);
ITTNOTIFY_EXPORT void __itt_heap_free_begin(__itt_heap_function, void*);
ITTNOTIFY_EXPORT void __itt_heap_free_end(__itt_heap_function, void*);
ITTNOTIFY_EXPORT void __itt_heap_reallocate_begin(__itt_heap_function, void*,
std::size_t, int);
ITTNOTIFY_EXPORT void __itt_heap_reallocate_end(__itt_heap_function, void*,
void**, std::size_t, int);
ITTNOTIFY_EXPORT void __itt_heap_internal_access_begin();
ITTNOTIFY_EXPORT void __itt_heap_internal_access_end();

ITTNOTIFY_EXPORT __itt_mark_type __itt_mark_create(char const* name);
ITTNOTIFY_EXPORT int __itt_mark_off(__itt_mark_type mark);
ITTNOTIFY_EXPORT int __itt_mark(__itt_mark_type mark, char const* par);
ITTNOTIFY_EXPORT __itt_caller __itt_stack_caller_create();
ITTNOTIFY_EXPORT void __itt_stack_callee_enter(__itt_caller ctx);
ITTNOTIFY_EXPORT void __itt_stack_callee_leave(__itt_caller ctx);
ITTNOTIFY_EXPORT void __itt_stack_caller_destroy(__itt_caller ctx);

ITTNOTIFY_EXPORT extern __itt_mark_type (*__itt_mark_create_ptr_)(char const*
name);
ITTNOTIFY_EXPORT extern int (*__itt_mark_off_ptr_)(__itt_mark_type mark);
ITTNOTIFY_EXPORT extern int (*__itt_mark_ptr_)(__itt_mark_type mark, char
const* par);
ITTNOTIFY_EXPORT extern __itt_caller (*__itt_stack_callee_create_ptr_)();
ITTNOTIFY_EXPORT extern void (*__itt_stack_caller_enter_ptr_)(__itt_caller ctx);
ITTNOTIFY_EXPORT extern void (*__itt_stack_callee_leave_ptr_)(__itt_caller ctx);
ITTNOTIFY_EXPORT extern void (*__itt_stack_caller_destroy_ptr_)(__itt_caller
ctx);

#ifdef __cplusplus
}
#endif

// make a bunch of function pointers
#define __itt_sync_create_ptr __itt_sync_create
#define __itt_frame_begin_v3_ptr __itt_frame_begin_v3
#define __itt_frame_end_v3_ptr __itt_frame_end_v3
#define __itt_id_create_ptr __itt_id_create
#define __itt_id_create_ex_ptr __itt_id_create_ex
#define __itt_id_destroy_ptr __itt_id_destroy
#define __itt_id_destroy_ex_ptr __itt_id_destroy_ex
#define __itt_id_make_ptr __itt_id_make
#define __itt_metadata_add_ptr __itt_metadata_add
#define __itt_metadata_add_with_scope_ptr __itt_metadata_add_with_scope
#define __itt_metadata_str_add_ptr __itt_metadata_str_add
#define __itt_metadata_str_add_with_scope_ptr __itt_metadata_str_add_with_scope
#define __itt_relation_add_ptr __itt_relation_add
#define __itt_relation_add_to_current_ptr __itt_relation_add_to_current
#define __itt__set_track_ptr __itt__set_track
#define __itt_string_handle_create_ptr __itt_string_handle_create
#define __itt_task_begin_ptr __itt_task_begin
#define __itt_task_begin_ex_ptr __itt_task_begin_ex
#define __itt_task_begin_fn_ptr __itt_task_begin_fn
#define __itt_task_end_ptr __itt_task_end
#define __itt_task_end_ex_ptr __itt_task_end_ex
#define __itt_task_group_ptr __itt_task_group
#define __itt_thread_set_name_ptr __itt_thread_set_name
#define __itt_track_group_create_ptr __itt_track_group_create
#define __itt_track_create_ptr __itt_track_create
#define __itt_domain_create_ptr __itt_domain_create

#define __itt_sync_create_ptr __itt_sync_create
#define __itt_sync_rename_ptr __itt_sync_rename
#define __itt_sync_prepare_ptr __itt_sync_prepare
#define __itt_sync_acquired_ptr __itt_sync_acquired
#define __itt_sync_cancel_ptr __itt_sync_cancel
#define __itt_sync_releasing_ptr __itt_sync_releasing
#define __itt_sync_destroy_ptr __itt_sync_destroy
#define __itt_thread_ignore_ptr __itt_thread_ignore
#define __itt_heap_function_create_ptr __itt_heap_function_create
#define __itt_heap_allocate_begin_ptr __itt_heap_allocate_begin
#define __itt_heap_allocate_end_ptr __itt_heap_allocate_end
#define __itt_heap_free_begin_ptr __itt_heap_free_begin
#define __itt_heap_free_end_ptr __itt_heap_free_end
#define __itt_heap_reallocate_begin_ptr __itt_heap_reallocate_begin
#define __itt_heap_reallocate_end_ptr __itt_heap_reallocate_end
#define __itt_heap_internal_access_begin_ptr __itt_heap_internal_access_begin
#define __itt_heap_internal_access_end_ptr __itt_heap_internal_access_end

#if 0
#define __itt_mark_create_ptr __itt_mark_create
#define __itt_mark_off_ptr __itt_mark_off
#define __itt_mark_ptr __itt_mark
#define __itt_stack_caller_destroy_ptr __itt_stack_caller_destroy
#define __itt_stack_callee_leave_ptr __itt_stack_callee_leave
#define __itt_stack_callee_enter_ptr __itt_stack_callee_enter
#define __itt_stack_caller_create_ptr __itt_stack_caller_create
#endif

#endif // ITTNOTIFY_H
