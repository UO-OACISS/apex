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

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <memory_wrapper.h>
#include "apex_api.hpp"

#ifdef _MSC_VER
/* define these functions as non-intrinsic */
#pragma function( memcpy, strcpy, strcat )
#endif

/* At initializtion, we don't want to start tracking malloc/free
 * until both APEX is initialized AND the dynamic library loader
 * has finished initialization and is about to launch main.
 * So we use 2 flags to accomlpish this. */

bool& apex_ready() {
    static bool _ready = false;
    return _ready;
}

bool& dl_ready() {
    static bool _ready = true;
    return _ready;
}

bool& enabled() {
    static bool _enabled = true;
    return _enabled;
}

bool all_clear() {
    return apex_ready() && dl_ready() && enabled();
}

extern "C"
void apex_memory_initialized() {
    apex_memory_wrapper_init();
    apex_ready() = true;
}

extern "C"
void apex_memory_lights_out() {
    apex_ready() = false;
    static bool once{false};
    if (!once) {
        apex_report_leaks();
        once = true;
    }
}

extern "C"
void apex_memory_finalized() {
    apex_memory_lights_out();
}

extern "C"
void apex_memory_dl_initialized() {
    dl_ready() = true;
}

/* During startup, we need to do some memory management in case
 * malloc/free is called during the startup process. */

// Memory for bootstrapping.  must not be static!
char bootstrap_heap[BOOTSTRAP_HEAP_SIZE];
char * bootstrap_base = bootstrap_heap;

uintptr_t reportHeapLocation() {
    printf("Bootstrap heap located at: %p\n", &bootstrap_heap[0]);
    return (uintptr_t)&bootstrap_heap[0];
}

static inline int is_bootstrap(void * ptr) {
    char const * const p = (char*)ptr;
    return (p < bootstrap_heap + BOOTSTRAP_HEAP_SIZE) && (bootstrap_heap < p);
}

static void * bootstrap_alloc(size_t align, size_t size) {
    //static uintptr_t dummy = reportHeapLocation();
    //APEX_UNUSED(dummy);
    char * ptr;

    // Check alignment.  Default alignment is sizeof(long)
    if(!align) {
        align = sizeof(long);

        if (size < align) {
            // Align to the next lower power of two
            align = size;
            while (align & (align-1)) {
                align &= align-1;
            }
        }
    }

    // Calculate address
    ptr = (char*)(((size_t)bootstrap_base + (align-1)) & ~(align-1));
    bootstrap_base = ptr + size;

    // Check for overflow
    if (bootstrap_base >= (bootstrap_heap + BOOTSTRAP_HEAP_SIZE)) {
        // These calls are unsafe, but we're about to die anyway.
        printf("APEX bootstreap heap exceeded.  Increase BOOTSTRAP_HEAP_SIZE in " __FILE__ " and try again.\n");
        fflush(stdout);
        exit(1);
    }

    return (void*)ptr;
}

static inline void bootstrap_free(void * ptr) {
    // Do nothing: bootstrap memory is deallocated on program exit
    APEX_UNUSED(ptr);
}


#ifdef APEX_PRELOAD_LIB
/********************************/
/* LD_PRELOAD wrapper functions */
/********************************/

#define RESET_DLERROR() dlerror()
#define CHECK_DLERROR() { \
    char const * err = dlerror(); \
    if (err) { \
        printf("Error getting %s handle: %s\n", name, err); \
        fflush(stdout); \
        exit(1); \
    } \
}

template<class T> T
get_system_function_handle(char const * name, T caller)
{
    T handle;

    // Reset error pointer
    RESET_DLERROR();

    // Attempt to get the function handle
    handle = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));

    // Detect errors
    CHECK_DLERROR();

    // Prevent recursion if more than one wrapping approach has been loaded.
    // This happens because we support wrapping pthreads three ways at once:
    // #defines in Profiler.h, -Wl,-wrap on the link line, and LD_PRELOAD.
    if (handle == caller) {
        RESET_DLERROR();
        void * syms = dlopen(NULL, RTLD_NOW);
        CHECK_DLERROR();
        do {
            RESET_DLERROR();
            handle = reinterpret_cast<T>(dlsym(syms, name));
            CHECK_DLERROR();
        } while (handle == caller);
    }

    return handle;
}

extern "C"
void* malloc (size_t size) {
    static malloc_p _malloc = NULL;
    static bool initializing = false;
    static bool bootstrapped = false;
    if (!bootstrapped) {
        if (!initializing) {
            initializing = true;
            _malloc = get_system_function_handle<malloc_p>("malloc", &malloc);
        }
        if (!_malloc) {
            return bootstrap_alloc(0, size);
        }
        if (!all_clear()) {
            return _malloc(size);
        }
        bootstrapped = true;
    }
    if (all_clear()) {
        return apex_malloc_wrapper(_malloc, size);
    }
    return _malloc(size);
}

extern "C"
void free (void* ptr) {
    static free_p _free = NULL;
    static bool initializing = false;
    static bool bootstrapped = false;
    if (is_bootstrap(ptr)) {
        // do nothing, effectively
        return bootstrap_free(ptr);
    }
    if (!bootstrapped) {
        if (!initializing) {
            initializing = true;
            _free = get_system_function_handle<free_p>("free", &free);
        }
        if (!_free) {
            // do nothing, effectively
            return bootstrap_free(ptr);
        }
        if (!all_clear()) {
            return _free(ptr);
        }
        bootstrapped = true;
    }
    if (all_clear()) {
        return apex_free_wrapper(_free, ptr);
    }
    return _free(ptr);
}

extern "C"
int puts (const char* s) {
    static puts_p _puts = NULL;
    static bool initializing = false;
    static bool bootstrapped = false;
    if (!bootstrapped) {
        if (!initializing) {
            initializing = true;
            _puts = get_system_function_handle<puts_p>("puts", &puts);
        }
        if (!_puts) {
            // do nothing, effectively
            return 0;
        }
        bootstrapped = true;
    }
    enabled() = false;
    auto r = _puts(s);
    enabled() = true;
    return r;
}

extern "C"
void* calloc (size_t nmemb, size_t size) {
    static calloc_p _calloc = NULL;
    static bool initializing = false;
    static bool bootstrapped = false;
    if (!bootstrapped) {
        if (!initializing) {
            initializing = true;
            _calloc = get_system_function_handle<calloc_p>("calloc", &calloc);
        }
        if (!_calloc) {
            return bootstrap_alloc(0, (nmemb*size));
        }
        if (!all_clear()) {
            return _calloc(nmemb, size);
        }
        bootstrapped = true;
    }
    if (all_clear()) {
        return apex_calloc_wrapper(_calloc, nmemb, size);
    }
    return _calloc(nmemb, size);
}

extern "C"
void* realloc (void* ptr, size_t size) {
    static realloc_p _realloc = NULL;
    static bool initializing = false;
    static bool bootstrapped = false;
    if (!bootstrapped) {
        if (!initializing) {
            initializing = true;
            _realloc = get_system_function_handle<realloc_p>("realloc", &realloc);
        }
        if (!_realloc) {
            return bootstrap_alloc(0, size);
        }
        if (!all_clear()) {
            return _realloc(ptr, size);
        }
        bootstrapped = true;
    }
    if (all_clear()) {
        return apex_realloc_wrapper(_realloc, ptr, size);
    }
    return _realloc(ptr, size);
}

#if 0
#if defined(memalign)
void* memalign (size_t alignment, size_t size) {
    static memalign_p _memalign = NULL;
    if (!_memalign) {
        _memalign = get_system_function_handle<memalign_p>("memalign", &memalign);
    }
    return apex_memalign_wrapper(_memalign, alignment, size);
}
#endif

#if defined(reallocarray)
void* reallocarray (void* ptr, size_t nmemb, size_t size) {
    static reallocarray_p _reallocarray = NULL;
    if (!_reallocarray) {
        _reallocarray = get_system_function_handle<reallocarray_p>("reallocarray", &reallocarray);
    }
    return apex_reallocarray_wrapper(_reallocarray, ptr, nmemb, size);
}
#endif

#if defined(reallocf)
void* reallocf (void* ptr, size_t size) {
    static reallocf_p _reallocf = NULL;
    if (!_reallocf) {
        _reallocf = get_system_function_handle<reallocf_p>("reallocf", &reallocf);
    }
    return apex_reallocf_wrapper(_reallocf, ptr, size);
}
#endif

#if defined(valloc)
void* valloc (size_t size) {
    static valloc_p _valloc = NULL;
    if (!_valloc) {
        _valloc = get_system_function_handle<valloc_p>("valloc", &valloc);
    }
    return apex_valloc_wrapper(_valloc, size);
}
#endif

#if defined(malloc_usable_size)
size_t malloc_usable_size (void* ptr) {
    static malloc_usable_size_p _malloc_usable_size = NULL;
    if (!_malloc_usable_size) {
        _malloc_usable_size = get_system_function_handle<malloc_usable_size_p>("malloc_usable_size", &malloc_usable_size);
    }
    return apex_malloc_usable_size_wrapper(_malloc_usable_size, ptr);
}
#endif

#endif

#else // Wrap via the the link line.

void* __real_malloc(size_t);
void* __wrap_malloc(size_t size) {
    return apex_malloc_wrapper(__real_malloc, size);
}

void __real_free(void*);
void __wrap_free(void* ptr) {
    return apex_free_wrapper(__real_free, ptr);
}

void* __real_calloc(size_t, size_t);
void* __wrap_calloc(size_t nmemb, size_t size) {
    return apex_calloc_wrapper(__real_calloc, nmemb, size);
}

void* __real_realloc(void*, size_t);
void* __wrap_realloc(void* ptr, size_t size) {
    return apex_realloc_wrapper(__real_realloc, ptr, size);
}

#if 0
#if defined(memalign)
void* __real_memalign(size_t, size_t);
void* __wrap_memalign(size_t alignment, size_t size) {
    return apex_memalign_wrapper(__real_memalign, alignment, size);
}
#endif

#if defined(reallocarray)
void* __real_reallocarray(void*, size_t, size_t);
void* __wrap_reallocarray(void* ptr, size_t nmemb, size_t size) {
    return apex_reallocarray_wrapper(__real_reallocarray, ptr, nmemb, size);
}
#endif

#if defined(reallocf)
void* __real_reallocf(void*, size_t);
void* __wrap_reallocf(void* ptr, size_t size) {
    return apex_reallocf_wrapper(__real_reallocf, ptr, size);
}
#endif

#if defined(valloc)
void* __real_valloc(size_t);
void* __wrap_valloc(size_t size) {
    return apex_valloc_wrapper(__vallocllocf, size);
}
#endif

#if defined(malloc_usable_size)
size_t __real_malloc_usable_size(void*);
size_t __wrap_malloc_usable_size(void* ptr) {
    return apex_malloc_usable_size_wrapper(__malloc_usable_size, ptr);
}
#endif
#endif

#endif //APEX_PRELOAD_LIB
