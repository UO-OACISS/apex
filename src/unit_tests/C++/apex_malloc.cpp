#include <stdlib.h>
#include <iostream>
#include "apex_api.hpp"

void bar(char* data) {
    apex::scoped_timer(__func__);
    auto s = strlen(data);
    std::cout << "Size: " << s << std::endl;
}

void test_malloc() {
    apex::scoped_timer(__func__);
    auto foo = (char*)(malloc(42 * sizeof(char)));
    memset(foo, 'a', 41);
    foo[41] = 0;
    bar(foo);
    free(foo);
}

void test_calloc() {
    apex::scoped_timer(__func__);
    auto foo = (char*)(calloc(42, sizeof(char)));
    memset(foo, 'a', 41);
    bar(foo);
    free(foo);
}

void test_realloc() {
    apex::scoped_timer(__func__);
    auto foo = (char*)(malloc(42 * sizeof(char)));
    memset(foo, 'a', 41);
    foo[41] = 0;
    bar(foo);
    foo = (char*)(realloc(foo, 84 * sizeof(char)));
    memset(foo, 'a', 83);
    foo[83] = 0;
    bar(foo);
    free(foo);
}

void test_all(void) {
  test_malloc();
  test_calloc();
  test_realloc();
}

void apex_enable_memory_wrapper(void);
void apex_disable_memory_wrapper(void);

int main (int argc, char** argv) {
  APEX_UNUSED(argc);
  APEX_UNUSED(argv);
  apex::init("apex memory wrapper unit test", 0, 1);
  apex::apex_options::use_screen_output(true);
  apex::apex_options::track_cpu_memory(true);
  test_all();
  apex::enable_memory_wrapper();
  test_all();
  apex::disable_memory_wrapper();
  apex::finalize();
  apex::cleanup();
  return 0;
}

