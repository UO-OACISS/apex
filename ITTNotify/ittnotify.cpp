#include "ittnotify.h"
#include <stdlib.h>
#include <string.h>
#include "apex.hpp"

#if 0
#include <iostream>
#define APEX_TRACER {cout << __FUNCTION__ << " ["<< __FILE__ << ":" << __LINE__ << "]" << endl;}
#else
#define APEX_TRACER 
#endif

//#ifdef __cplusplus
//extern "C" {
//#endif
void __itt_frame_begin_v3(__itt_domain const* frame, __itt_id* id) { APEX_TRACER }
void __itt_frame_end_v3(__itt_domain const* frame, __itt_id* id) { APEX_TRACER }
void __itt_id_create(__itt_domain const*, __itt_id id) { APEX_TRACER }
void __itt_id_create_ex(__itt_domain const*, 
     __itt_clock_domain* clock_domain, UINT64 timestamp, __itt_id* id) { APEX_TRACER }
void __itt_id_destroy(const __itt_domain * domain, __itt_id* id) { APEX_TRACER }
void __itt_id_destroy_ex(__itt_domain* domain, 
     __itt_clock_domain* clock_domain, UINT64 timestamp, __itt_id* id) { APEX_TRACER }
__itt_id __itt_id_make(void*, unsigned long) { APEX_TRACER return __itt_null; }
void __itt_metadata_add (const __itt_domain * domain,
      __itt_id id, __itt_string_handle * name, __itt_marker_scope scope) { APEX_TRACER }
void _itt_metadata_add_with_scope(const __itt_domain * domain, 
      __itt_scope scope, __itt_string_handle *key, __itt_metadata_type type,
      size_t count, void *data) { APEX_TRACER }
void __itt_metadata_str_add (const __itt_domain * domain,
      __itt_id id, __itt_string_handle * key, const char * data, size_t length) { APEX_TRACER }
void _itt_metadata_str_add_with_scope(const __itt_domain * domain, 
      __itt_scope scope, __itt_string_handle *key, const char * data, size_t length) { APEX_TRACER }
void __itt_relation_add (const __itt_domain * domain, __itt_id head, 
      __itt_relation relation, __itt_id tail) { APEX_TRACER }
void __itt_relation_add_to_current (const __itt_domain * domain, 
      __itt_relation relation, __itt_id tail) { APEX_TRACER }
void __itt__set_track(__itt_track * track) { APEX_TRACER }
void __itt_task_begin_ex(__itt_domain* domain, __itt_clock_domain* clock_domain, 
      UINT64 timestamp, __itt_id id, __itt_id parentid, __itt_string_handle *name) { APEX_TRACER }
void __itt_task_begin_fn(const __itt_domain * domain, __itt_id taskid, 
      __itt_id parentid, void * fn) { APEX_TRACER }
void __itt_task_end_ex(__itt_domain const*) { APEX_TRACER }
void __itt_task_group(const __itt_domain * domain, __itt_id id, 
      __itt_id parentid, __itt_string_handle * name) { APEX_TRACER }
__itt_track_group* __itt_track_group_create(__itt_string_handle* name, 
      __itt_track_group_type type) { APEX_TRACER return NULL; }
__itt_track* __itt_track_create(__itt_track_group* track_group, 
      __itt_string_handle* name, __itt_track_type track_type) { APEX_TRACER return NULL; }

__itt_domain* __itt_domain_create(char const* name) { 
  APEX_TRACER
  __itt_domain *domain = (__itt_domain*)(malloc(sizeof(__itt_domain)));
  domain->nameA = strdup(name);
  apex::init(NULL);
  apex::set_node_id(0);
  return domain;
}

__itt_string_handle* __itt_string_handle_create(char const* name) {
  APEX_TRACER
  __itt_string_handle *string_handle = (__itt_string_handle*)(malloc(sizeof(__itt_string_handle)));
  string_handle->strA = strdup(name);
  return string_handle;
}

void __itt_task_begin(__itt_domain const* domain, __itt_id taskid, __itt_id parentid, __itt_string_handle* handle) {
  APEX_TRACER
  std::string tmp = string(handle->strA);
  apex::start(tmp);
}
void __itt_task_end(__itt_domain const*) {
  APEX_TRACER
  apex::stop(NULL);
}

void __itt_thread_set_name (const char * name) {
  APEX_TRACER
  apex::register_thread(string(name));
}

void __itt_sync_create(void*, const char*, const char*, int) { APEX_TRACER };
void __itt_sync_rename(void*, const char*) { APEX_TRACER };
void __itt_sync_prepare(void*) { APEX_TRACER };
void __itt_sync_acquired(void*) { APEX_TRACER };
void __itt_sync_cancel(void*) { APEX_TRACER };
void __itt_sync_releasing(void*) { APEX_TRACER };
void __itt_sync_destroy(void*) { APEX_TRACER };
void __itt_thread_ignore() { APEX_TRACER };
__itt_heap_function __itt_heap_function_create(const char*, const char*) { APEX_TRACER return NULL; };
void __itt_heap_allocate_begin(__itt_heap_function, std::size_t, int) { APEX_TRACER };
void __itt_heap_allocate_end(__itt_heap_function, void**, std::size_t, int) { APEX_TRACER };
void __itt_heap_free_begin(__itt_heap_function, void*) { APEX_TRACER };
void __itt_heap_free_end(__itt_heap_function, void*) { APEX_TRACER };
void __itt_heap_reallocate_begin(__itt_heap_function, void*, std::size_t, int) { APEX_TRACER };
void __itt_heap_reallocate_end(__itt_heap_function, void*, void**, std::size_t, int) { APEX_TRACER };
void __itt_heap_internal_access_begin() { APEX_TRACER };
void __itt_heap_internal_access_end() { APEX_TRACER };

__itt_mark_type __itt_mark_create(char const* name) { APEX_TRACER return 0; };
int __itt_mark_off(__itt_mark_type mark) { APEX_TRACER return 0; };
int __itt_mark(__itt_mark_type mark, char const* par) { APEX_TRACER return 0; };
__itt_caller __itt_stack_caller_create() { APEX_TRACER return NULL; };
void __itt_stack_callee_enter(__itt_caller ctx) { APEX_TRACER };
void __itt_stack_callee_leave(__itt_caller ctx) { APEX_TRACER };
void __itt_stack_caller_destroy(__itt_caller ctx) { APEX_TRACER };

//#ifdef __cplusplus
//}
//#endif

// assign some function pointers
ITTNOTIFY_EXPORT __itt_mark_type (*__itt_mark_create_ptr_)(char const* name) = __itt_mark_create;
ITTNOTIFY_EXPORT int (*__itt_mark_off_ptr_)(__itt_mark_type mark) = __itt_mark_off;
ITTNOTIFY_EXPORT int (*__itt_mark_ptr_)(__itt_mark_type mark, char const* par) = __itt_mark;
ITTNOTIFY_EXPORT void (*__itt_stack_callee_enter_ptr_)(__itt_caller ctx) = __itt_stack_callee_enter;
ITTNOTIFY_EXPORT void (*__itt_stack_callee_leave_ptr_)(__itt_caller ctx) = __itt_stack_callee_leave;
ITTNOTIFY_EXPORT __itt_caller (*__itt_stack_caller_create_ptr_)() = __itt_stack_caller_create;
ITTNOTIFY_EXPORT void (*__itt_stack_caller_destroy_ptr_)(__itt_caller ctx) = __itt_stack_caller_destroy;







