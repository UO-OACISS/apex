#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <memory_wrapper.h>

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

static
void * get_system_function_handle(char const * name, void * caller)
{
  void * handle;

  // Reset error pointer
  RESET_DLERROR();

  // Attempt to get the function handle
  handle = dlsym(RTLD_NEXT, name);

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
      handle = dlsym(syms, name);
      CHECK_DLERROR();
    } while (handle == caller);
  }

  return handle;
}

void* malloc (size_t size) {
  static malloc_p _malloc = NULL;
  if (!_malloc) {
    _malloc = (malloc_p)get_system_function_handle("malloc", (void*)malloc);
  }
  return apex_malloc_wrapper(_malloc, size);
}

void* calloc (size_t nmemb, size_t size) {
  static calloc_p _calloc = NULL;
  if (!_calloc) {
    _calloc = (calloc_p)get_system_function_handle("calloc", (void*)calloc);
  }
  return apex_calloc_wrapper(_calloc, nmemb, size);
}

#if defined(memalign)
void* memalign (size_t alignment, size_t size) {
  static memalign_p _memalign = NULL;
  if (!_memalign) {
    _memalign = (memalign_p)get_system_function_handle("memalign", (void*)memalign);
  }
  return apex_memalign_wrapper(_memalign, alignment, size);
}
#endif

void* realloc (void* ptr, size_t size) {
  static realloc_p _realloc = NULL;
  if (!_realloc) {
    _realloc = (realloc_p)get_system_function_handle("realloc", (void*)realloc);
  }
  return apex_realloc_wrapper(_realloc, ptr, size);
}

#if defined(reallocarray)
void* reallocarray (void* ptr, size_t nmemb, size_t size) {
  static reallocarray_p _reallocarray = NULL;
  if (!_reallocarray) {
    _reallocarray = (reallocarray_p)get_system_function_handle("reallocarray", (void*)reallocarray);
  }
  return apex_reallocarray_wrapper(_reallocarray, ptr, nmemb, size);
}
#endif

#if defined(reallocf)
void* reallocf (void* ptr, size_t size) {
  static reallocf_p _reallocf = NULL;
  if (!_reallocf) {
    _reallocf = (reallocf_p)get_system_function_handle("reallocf", (void*)reallocf);
  }
  return apex_reallocf_wrapper(_reallocf, ptr, size);
}
#endif

#if defined(valloc)
void* valloc (size_t size) {
  static valloc_p _valloc = NULL;
  if (!_valloc) {
    _valloc = (valloc_p)get_system_function_handle("valloc", (void*)valloc);
  }
  return apex_valloc_wrapper(_valloc, size);
}
#endif

#if defined(malloc_usable_size)
size_t malloc_usable_size (void* ptr) {
  static malloc_usable_size_p _malloc_usable_size = NULL;
  if (!_malloc_usable_size) {
    _malloc_usable_size = (malloc_usable_size_p)get_system_function_handle("malloc_usable_size", (void*)malloc_usable_size);
  }
  return apex_malloc_usable_size_wrapper(_malloc_usable_size, ptr);
}
#endif

void free (void* ptr) {
  static free_p _free = NULL;
  if (!_free) {
    _free = (free_p)get_system_function_handle("free", (void*)free);
  }
  return apex_free_wrapper(_free, ptr);
}

#else // Wrap via the the link line.

void* __real_malloc(size_t);
void* __wrap_malloc(size_t size) {
  return apex_malloc_wrapper(__real_malloc, size);
}

void* __real_calloc(size_t, size_t);
void* __wrap_calloc(size_t nmemb, size_t size) {
  return apex_calloc_wrapper(__real_calloc, nmemb, size);
}

#if defined(memalign)
void* __real_memalign(size_t, size_t);
void* __wrap_memalign(size_t alignment, size_t size) {
  return apex_memalign_wrapper(__real_memalign, alignment, size);
}
#endif

void* __real_realloc(void*, size_t);
void* __wrap_realloc(void* ptr, size_t size) {
  return apex_realloc_wrapper(__real_realloc, ptr, size);
}

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

void __real_free(void*);
void __wrap_free(void* ptr) {
  return apex_free_wrapper(__real_free, ptr);
}

#endif //APEX_PRELOAD_LIB
