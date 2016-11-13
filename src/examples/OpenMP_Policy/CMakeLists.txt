# Copyright (c) 2015 University of Oregon
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Make sure the compiler can find include files from our Apex library. 
include_directories (. ${APEX_SOURCE_DIR}/src/apex ${APEX_SOURCE_DIR}/rapidjson/include)

# Make sure the linker can find the Apex library once it is built. 
link_directories (${APEX_BINARY_DIR}/src/apex)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_library(apex_openmp_policy SHARED apex_openmp_policy.cpp)
add_dependencies (apex_openmp_policy apex)
target_link_libraries(apex_openmp_policy apex ${LIBS})
set_target_properties(apex_openmp_policy PROPERTIES OUTPUT_NAME apex_openmp_policy)

add_executable (policy_test policy_test.cpp)
add_dependencies (policy_test apex)

INSTALL(TARGETS apex_openmp_policy policy_test
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

