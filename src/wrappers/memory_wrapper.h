#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <memory.h>

typedef void* (*malloc_p)(size_t);
typedef void* (*calloc_p)(size_t, size_t);
#if defined(memalign)
typedef void* (*memalign_p)(void*, size_t, size_t);
#endif
typedef void* (*realloc_p)(void*, size_t);
#if defined(reallocarray)
typedef void* (*reallocarray_p)(void*, size_t, size_t);
#endif
#if defined(reallocf)
typedef void* (*reallocf_p)(void*, size_t);
#endif
#if defined(valloc)
typedef void* (*valloc_p)(size_t);
#endif
#if defined(malloc_usable_size)
typedef size_t (*valloc_p)(void*);
#endif
typedef void  (*free_p)(void*);

#ifdef __cplusplus
extern "C" {
#endif

void* apex_malloc_wrapper(malloc_p malloc_call, size_t size);
void* apex_calloc_wrapper(calloc_p calloc_call, size_t nmemb, size_t size);
#if defined(memalign)
void* apex_memalign_wrapper(memalign_p calloc_call, size_t align, size_t size);
#endif
void* apex_realloc_wrapper(realloc_p realloc_call, void* ptr, size_t size);
#if defined(reallocarray)
void* apex_reallocarray_wrapper(reallocarray_p reallocarray_call, void* ptr, size_t nmemb, size_t size);
#endif
#if defined(reallocf)
void* apex_reallocf_wrapper(reallocf_p reallocf_call, void* ptr, size_t size);
#endif
#if defined(valloc)
void* apex_valloc_wrapper(valloc_p valloc_call, size_t size);
#endif
#if defined(malloc_usable_size)
void* apex_malloc_usable_size_wrapper(malloc_usable_size_p malloc_usable_size_call, void* ptr);
#endif
void  apex_free_wrapper(free_p free_call, void* ptr);

#ifdef __cplusplus
}
#endif

#define APEX_PRELOAD_LIB
