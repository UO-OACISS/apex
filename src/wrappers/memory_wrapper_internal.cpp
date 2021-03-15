#include "apex_api.hpp"
#include "memory_wrapper.h"
#include <memory>

///////////////////////////////////////////////////////////////////////////////
// Below is the malloc wrapper
///////////////////////////////////////////////////////////////////////////////

bool& inWrapper() {
    thread_local static bool _inWrapper = false;
    return _inWrapper;
}

extern "C"
void* apex_malloc_wrapper(malloc_p malloc_call, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return malloc_call(size);
        printf("Here!\n");
    } else {
        inWrapper() = true;
        printf("Here!\n");

        // do the allocation
        auto retval = malloc_call(size);
        apex::sample_value("malloc bytes", size, true);

        inWrapper() = false;
        return retval;
    }
}

extern "C"
void* apex_calloc_wrapper(calloc_p calloc_call, size_t nmemb, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return calloc_call(nmemb, size);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = calloc_call(nmemb, size);

        inWrapper() = false;
        return retval;
    }
}

#if defined(memalign)
extern "C"
void* apex_memalign_wrapper(memalign_p memalign_call, size_t nmemb, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return memalign_call(nmemb, size);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = memalign_call(nmemb, size);

        inWrapper() = false;
        return retval;
    }
}
#endif

extern "C"
void* apex_realloc_wrapper(realloc_p realloc_call, void* ptr, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return realloc_call(ptr, size);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = realloc_call(ptr, size);

        inWrapper() = false;
        return retval;
    }
}

#if defined(reallocarray)
extern "C"
void* apex_reallocarray_wrapper(reallocarray_p reallocarray_call, void* ptr, size_t nmemb, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return reallocarray_call(ptr, nmemb, size);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = reallocarray_call(ptr, nmemb, size);

        inWrapper() = false;
        return retval;
    }
}
#endif

#if defined(reallocf)
extern "C"
void* apex_reallocf_wrapper(reallocf_p reallocf_call, void* ptr, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return reallocf_call(ptr, size);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = reallocf_call(ptr, size);

        inWrapper() = false;
        return retval;
    }
}
#endif

#if defined(valloc)
extern "C"
void* apex_valloc_wrapper(valloc_p valloc_call, size_t size) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return valloc_call(size);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = valloc_call(size);

        inWrapper() = false;
        return retval;
    }
}
#endif

#if defined(malloc_usable_size)
extern "C"
size_t apex_malloc_usable_size_wrapper(malloc_usable_size_p malloc_usable_size_call, void* ptr) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return malloc_usable_size_call(ptr);
    } else {
        inWrapper() = true;

        // do the allocation
        auto retval = malloc_usable_size_call(ptr);

        inWrapper() = false;
        return retval;
    }
}
#endif

extern "C"
void apex_free_wrapper(free_p free_call, void* ptr) {
    if(inWrapper()) {
        // Another wrapper has already intercepted the call so just pass through
        return free_call(ptr);
    } else {
        inWrapper() = true;

        // do the allocation
        free_call(ptr);

        inWrapper() = false;
        return;
    }
}

extern "C" void* apex_malloc(size_t size) {
  return apex_malloc_wrapper(malloc, size);
}

extern "C" void* apex_calloc(size_t nmemb, size_t size) {
  return apex_calloc_wrapper(calloc, nmemb, size);
}

#if defined(memalign)
extern "C" void* apex_memalign(size_t nmemb, size_t size) {
  return apex_memalign_wrapper(memalign, nmemb, size);
}
#endif

extern "C" void* apex_realloc(void* ptr, size_t size) {
  return apex_realloc_wrapper(realloc, ptr, size);
}

#if defined(reallocarray)
extern "C" void* apex_reallocarray(void* ptr, size_t nmemb, size_t size) {
  return apex_reallocarray_wrapper(reallocarray, ptr, nmemb, size);
}
#endif

#if defined(reallocf)
extern "C" void* apex_reallocf(void* ptr, size_t size) {
  return apex_reallocf_wrapper(reallocf, ptr, size);
}
#endif

#if defined(valloc)
extern "C" void* apex_valloc(size_t size) {
  return apex_valloc_wrapper(valloc, size);
}
#endif

#if defined(malloc_usable_size)
extern "C" void* apex_malloc_usable_size(void* ptr) {
  return apex_malloc_usable_size_wrapper(malloc_usable_size, ptr);
}
#endif

extern "C" void apex_free(void* ptr) {
  return apex_free_wrapper(free, ptr);
}


