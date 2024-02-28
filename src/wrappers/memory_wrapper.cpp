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
#include "memory_wrapper.hpp"

#ifdef _MSC_VER
/* define these functions as non-intrinsic */
#pragma function( memcpy, strcpy, strcat )
#endif

/* At initializtion, we don't want to start tracking malloc/free
 * until both APEX is initialized AND the dynamic library loader
 * has finished initialization and is about to launch main.
 * So we use 2 flags to accomlpish this. */

bool& apex_memory_ready() {
    static bool _ready = false;
    return _ready;
}

bool& apex_dl_ready() {
    static bool _ready = true;
    return _ready;
}

bool& apex_memory_enabled() {
    static bool _enabled = true;
    return _enabled;
}

bool apex_memory_all_clear() {
    return apex_memory_ready() && apex_dl_ready() && apex_memory_enabled();
}

extern "C"
void apex_memory_initialized() {
    apex_memory_wrapper_init();
    apex_memory_ready() = true;
}

extern "C"
void apex_memory_lights_out() {
    apex_memory_ready() = false;
    apex::apex_report_leaks();
}

extern "C"
void apex_memory_finalized() {
    apex_memory_lights_out();
}

extern "C"
void apex_memory_dl_initialized() {
    apex_dl_ready() = true;
}

/* During startup, we need to do some memory management in case
 * malloc/free is called during the startup process. */

// Memory for bootstrapping.  must not be static!
char apex_memory_bootstrap_heap[BOOTSTRAP_HEAP_SIZE];
char * apex_memory_bootstrap_base = apex_memory_bootstrap_heap;

/*
uintptr_t apex_memory_reportHeapLocation() {
    printf("Bootstrap heap located at: %p\n", (void*)(&apex_memory_bootstrap_heap[0]));
    return (uintptr_t)&apex_memory_bootstrap_heap[0];
}
*/

static inline int apex_memory_is_bootstrap(void * ptr) {
    char const * const p = (char*)ptr;
    return (p < apex_memory_bootstrap_heap + BOOTSTRAP_HEAP_SIZE) && (apex_memory_bootstrap_heap < p);
}

static void * apex_memory_bootstrap_alloc(size_t align, size_t size) {
    //static uintptr_t dummy = apex_memory_reportHeapLocation();
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
    ptr = (char*)(((size_t)apex_memory_bootstrap_base + (align-1)) & ~(align-1));
    apex_memory_bootstrap_base = ptr + size;

    // Check for overflow
    if (apex_memory_bootstrap_base >= (apex_memory_bootstrap_heap + BOOTSTRAP_HEAP_SIZE)) {
        // These calls are unsafe, but we're about to die anyway.
        printf("APEX bootstreap heap exceeded.  Increase BOOTSTRAP_HEAP_SIZE in " __FILE__ " and try again.\n");
        fflush(stdout);
        exit(1);
    }

    return (void*)ptr;
}

static inline void apex_memory_bootstrap_free(void * ptr) {
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
apex_get_system_function_handle(char const * name, T caller)
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

// __THROW is defined by gcc extensions to allow C and C++
// compilers to build with the same headers. If it isn't 
// defined, make sure we define it.

#ifndef __THROW
#define __THROW
#endif

extern "C"
void* malloc (size_t size) __THROW {
    static malloc_p _malloc = NULL;
    static bool initializing = false;
    static bool bootstrapped = false;
    if (!bootstrapped) {
        if (!initializing) {
            initializing = true;
            _malloc = apex_get_system_function_handle<malloc_p>("malloc", &malloc);
        }
        if (!_malloc) {
            return apex_memory_bootstrap_alloc(0, size);
        }
        if (!apex_memory_all_clear()) {
            return _malloc(size);
        }
        bootstrapped = true;
    }
    if (apex_memory_all_clear()) {
        return apex_malloc_wrapper(_malloc, size);
    }
    return _malloc(size);
}

extern "C"
void free (void* ptr) __THROW {
    static free_p _free = NULL;
    static bool initializing = false;
    static bool bootstrapped = false;
    if (apex_memory_is_bootstrap(ptr)) {
        // do nothing, effectively
        return apex_memory_bootstrap_free(ptr);
    }
    if (!bootstrapped) {
        if (!initializing) {
            initializing = true;
            _free = apex_get_system_function_handle<free_p>("free", &free);
        }
        if (!_free) {
            // do nothing, effectively
            return apex_memory_bootstrap_free(ptr);
        }
        if (!apex_memory_all_clear()) {
            return _free(ptr);
        }
        bootstrapped = true;
    }
    if (apex_memory_all_clear()) {
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
            _puts = apex_get_system_function_handle<puts_p>("puts", &puts);
        }
        if (!_puts) {
            // do nothing, effectively
            return 0;
        }
        bootstrapped = true;
    }
    apex_memory_enabled() = false;
    int r = _puts(s);
    apex_memory_enabled() = true;
    return r;
}

extern "C"
void* calloc (size_t nmemb, size_t size) __THROW {
    static calloc_p _calloc = NULL;
    static bool initializing = false;
    static bool bootstrapped = false;
    if (!bootstrapped) {
        if (!initializing) {
            initializing = true;
            _calloc = apex_get_system_function_handle<calloc_p>("calloc", &calloc);
        }
        if (!_calloc) {
            return apex_memory_bootstrap_alloc(0, (nmemb*size));
        }
        if (!apex_memory_all_clear()) {
            return _calloc(nmemb, size);
        }
        bootstrapped = true;
    }
    if (apex_memory_all_clear()) {
        return apex_calloc_wrapper(_calloc, nmemb, size);
    }
    return _calloc(nmemb, size);
}

extern "C"
void* realloc (void* ptr, size_t size) __THROW {
    static realloc_p _realloc = NULL;
    static bool initializing = false;
    static bool bootstrapped = false;
    if (!bootstrapped) {
        if (!initializing) {
            initializing = true;
            _realloc = apex_get_system_function_handle<realloc_p>("realloc", &realloc);
        }
        if (!_realloc) {
            return apex_memory_bootstrap_alloc(0, size);
        }
        if (!apex_memory_all_clear()) {
            return _realloc(ptr, size);
        }
        bootstrapped = true;
    }
    if (apex_memory_all_clear()) {
        return apex_realloc_wrapper(_realloc, ptr, size);
    }
    return _realloc(ptr, size);
}

#if 0
#if defined(memalign)
void* memalign (size_t alignment, size_t size) {
    static memalign_p _memalign = NULL;
    if (!_memalign) {
        _memalign = apex_get_system_function_handle<memalign_p>("memalign", &memalign);
    }
    return apex_memalign_wrapper(_memalign, alignment, size);
}
#endif

#if defined(reallocarray)
void* reallocarray (void* ptr, size_t nmemb, size_t size) {
    static reallocarray_p _reallocarray = NULL;
    if (!_reallocarray) {
        _reallocarray = apex_get_system_function_handle<reallocarray_p>("reallocarray", &reallocarray);
    }
    return apex_reallocarray_wrapper(_reallocarray, ptr, nmemb, size);
}
#endif

#if defined(reallocf)
void* reallocf (void* ptr, size_t size) {
    static reallocf_p _reallocf = NULL;
    if (!_reallocf) {
        _reallocf = apex_get_system_function_handle<reallocf_p>("reallocf", &reallocf);
    }
    return apex_reallocf_wrapper(_reallocf, ptr, size);
}
#endif

#if defined(valloc)
void* valloc (size_t size) {
    static valloc_p _valloc = NULL;
    if (!_valloc) {
        _valloc = apex_get_system_function_handle<valloc_p>("valloc", &valloc);
    }
    return apex_valloc_wrapper(_valloc, size);
}
#endif

#if defined(malloc_usable_size)
size_t malloc_usable_size (void* ptr) {
    static malloc_usable_size_p _malloc_usable_size = NULL;
    if (!_malloc_usable_size) {
        _malloc_usable_size = apex_get_system_function_handle<malloc_usable_size_p>("malloc_usable_size", &malloc_usable_size);
    }
    return apex_malloc_usable_size_wrapper(_malloc_usable_size, ptr);
}
#endif

#endif // if 0
#endif //APEX_PRELOAD_LIB
