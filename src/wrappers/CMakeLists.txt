# Make sure the compiler can find include files from our Apex library. 
include_directories (${APEX_SOURCE_DIR}/src/apex
${APEX_SOURCE_DIR}/src/wrappers)

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex)

# Add library called "testOverhead" that is built from the source file
add_library (apex_pthread_wrapper pthread_wrapper.c
pthread_wrapper_internal.cpp)
add_dependencies (apex_pthread_wrapper apex)

if(APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPTHREADS -lapex")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPTHREADS -lapex")
else()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -DPTHREADS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DPTHREADS -Wl,--no-as-needed -ldl")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -ldl")
endif()

INSTALL(FILES apex_exec DESTINATION bin
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
              GROUP_EXECUTE GROUP_READ
              WORLD_EXECUTE WORLD_READ)

INSTALL(TARGETS apex_pthread_wrapper 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
