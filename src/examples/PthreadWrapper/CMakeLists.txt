# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex) 

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex) 
link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper) 

set(sources apex_pthread_wrap.c)
source_group("Source Files" FILES ${sources})

# add example executable
add_executable(pthread_wrapper_example ${sources})
target_link_libraries (pthread_wrapper_example apex apex_pthread_wrapper)
add_dependencies (pthread_wrapper_example apex apex_pthread_wrapper)
add_test ("test_pthread_wrapper_example" pthread_wrapper_example)

if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPTHREADS")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -DPTHREADS")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -ldl")
    #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-wrap,pthread_create -Wl,-wrap,pthread_join -Wl,-wrap,pthread_exit -Wl,-wrap,pthread_barrier_wait")
endif()

