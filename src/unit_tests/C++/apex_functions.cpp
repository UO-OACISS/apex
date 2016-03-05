/**
 * Helpful web pages:
 * http://www.dreamincode.net/forums/topic/264061-c11-fun-with-functions/
 */

#include <functional>
#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include "apex_api.hpp"

template< class Function >
apex::profiler * apex_start_wrapper( Function func )
{
    if( func ) // if the call-wrapper has wrapped a callable object
    {
        int status;
        const char * name = abi::__cxa_demangle(typeid(func).name(), 0, 0, &status);
        std::cout << name << '\n';
        return apex::start(name);
    }
}

template< class Function >
apex::profiler * apex_start_lambda_wrapper( Function func )
{
    if( func ) // if the call-wrapper has wrapped a callable object
    {
        int status;
        const char * name = abi::__cxa_demangle(func.target_type().name(), 0, 0, &status);
        std::cout << name << '\n';
        return apex::start(name);
    }
}

/**
 * Lambdas have the "target_name" function defined.
 */
#define apex_start_lambda_wrapper_macro(__foo,__p) {\
    int __status; \
    const char * __name = abi::__cxa_demangle(__foo.target_type().name(), 0, 0, &__status); \
    std::cout << __name << '\n'; \
    __p = apex::start(__name); \
}
 
struct Foo {
    Foo(int num) : num_(num) {}
    void print_add(int i) const;
    int num_;
};

/*
// forward declarations
struct Foo; 
struct PrintNum;
// instantiate the templates
template apex::profiler * apex_start_wrapper(std::function<void (const Foo&, int)>);
template apex::profiler * apex_start_wrapper(std::function<void (const PrintNum&, int)>);
*/

void Foo::print_add(int i) const { 
    std::function<void(const Foo&, int)> f = &Foo::print_add;
    apex::profiler * p = apex_start_wrapper(f);
    std::cout << num_+i << '\n'; 
    apex::stop(p);
}
 
void print_num(int i)
{
    apex::profiler * p = apex::start((apex_function_address)(&print_num));
    std::cout << i << '\n';
    apex::stop(p);
}
 
struct PrintNum {
    void operator()(int i) const
    {
        //std::function<void(const PrintNum&, int)> f = &PrintNum::operator();
        //auto f = std::mem_fn(&PrintNum::operator());
        apex::profiler * p = apex_start_wrapper(&PrintNum::operator());
        std::cout << i << '\n';
        apex::stop(p);
    }
};
 
int main()
{
    apex::init("std::function test");
    // store a free function
    std::function<void(int)> f_display = print_num;
    f_display(-9);
 
    // store a lambda
    std::function<void()> f_display_42 = [&]() { 
        apex::profiler * p = apex_start_lambda_wrapper(f_display_42);
        print_num(42); 
        apex::stop(p);
    };
    f_display_42();
 
    // store another lambda, using the macro
    std::function<void()> f_display_24 = [&]() { 
        apex::profiler * p;
        apex_start_lambda_wrapper_macro(f_display_24,p);
        print_num(42); 
        apex::stop(p);
    };
    f_display_42();
 
    // store the result of a call to std::bind
    std::function<void()> f_display_31337 = std::bind(print_num, 31337);
    f_display_31337();
 
    // store a call to a member function
    std::function<void(const Foo&, int)> f_add_display = &Foo::print_add;
    const Foo foo(314159);
    f_add_display(foo, 1);
 
    // store a call to a member function and object
    using std::placeholders::_1;
    std::function<void(int)> f_add_display2= std::bind( &Foo::print_add, foo, _1 );
    f_add_display2(2);
 
    // store a call to a member function and object ptr
    std::function<void(int)> f_add_display3= std::bind( &Foo::print_add, &foo, _1 );
    f_add_display3(3);
 
    // store a call to a function object
    std::function<void(int)> f_display_obj = PrintNum();
    f_display_obj(18);
    apex::finalize();
    apex::cleanup();
}
