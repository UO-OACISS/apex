#  Copyright (c) 2014 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR})

# gcc on apple needs the apex library to link the dylib
# if(APPLE)
  link_directories (${APEX_BINARY_DIR})
# endif()

set(ITTNOTIFY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ITTNOTIFY_SOURCE_DIR ${ITTNOTIFY_SOURCE_DIR} PARENT_SCOPE)
set(ITTNOTIFY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(ITTNOTIFY_BINARY_DIR ${ITTNOTIFY_BINARY_DIR} PARENT_SCOPE)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(ittnofify_library_dependencies
      apex${CMAKE_DEBUG_POSTFIX} ${LIBS} ${Boost_LIBRARIES})
else()
  set(ittnofify_library_dependencies
      apex ${LIBS} ${Boost_LIBRARIES})
endif()

# Create a library called "ittnotify" which includes the source file
# "ittnotify.cpp".
# The extension is already found. Any number of sources could be listed here.
if(NOT MSVC)
  add_library (ittnotifyShared SHARED ittnotify.cpp)
  set_target_properties (ittnotifyShared PROPERTIES OUTPUT_NAME ittnotify)
  target_link_libraries (ittnotifyShared ${ittnofify_library_dependencies})
  add_dependencies (ittnotifyShared apexShared)
  set(ittnotify_install_targets ittnotifyShared)
endif()

# target_link_libraries (ittnotifyStatic ${LIBS} ${Boost_LIBRARIES})
# gcc on apple needs the apex library to link the dylib
# if(APPLE)
add_library (ittnotifyStatic STATIC ittnotify.cpp)
set_target_properties (ittnotifyStatic PROPERTIES OUTPUT_NAME ittnotify)
target_link_libraries (ittnotifyStatic ${ittnofify_library_dependencies})
add_dependencies (ittnotifyStatic apexStatic)
set(ittnotify_install_targets ${ittnotify_install_targets} ittnotifyStatic)
# else()
# target_link_libraries (ittnotifyShared ${LIBS})
# endif()

# add the binary tree to the search path for include files
# so that we will find ittnotify.h
include_directories ("${PROJECT_BINARY_DIR}/apex")

INSTALL(FILES ittnotify.h DESTINATION include)
INSTALL(FILES legacy/ittnotify.h DESTINATION include/legacy)

INSTALL(TARGETS ${ittnotify_install_targets}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib/hpx
  ARCHIVE DESTINATION lib/hpx
  #LIBRARY DESTINATION lib64
  #ARCHIVE DESTINATION lib64
)

if(NOT AMPLIFIER_ROOT)
  set(AMPLIFIER_ROOT ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
  set(HPX_HAVE_ITTNOTIFY ON PARENT_SCOPE)
endif()
