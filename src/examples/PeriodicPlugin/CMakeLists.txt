# Copyright (c) 2015 University of Oregon
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Make sure the compiler can find include files from our Apex library.
include_directories (. ${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)

add_library(apex_periodic_policy SHARED apex_periodic_policy.cpp)
add_dependencies (apex_periodic_policy apex)
target_link_libraries(apex_periodic_policy apex ${LIBS})
set_target_properties(apex_periodic_policy PROPERTIES OUTPUT_NAME apex_periodic_policy)

add_executable (periodic_policy_test policy_test.cpp)
add_dependencies (periodic_policy_test apex)
target_link_libraries (periodic_policy_test ${LIBS})

INSTALL(TARGETS apex_periodic_policy periodic_policy_test
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

