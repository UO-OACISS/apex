# The name of our project is "APEX". CMakeLists files in this project can 
# refer to the root source directory of the project as ${APEX_SOURCE_DIR} and 
# to the root binary directory of the project as ${APEX_BINARY_DIR}. 
cmake_minimum_required (VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

################################################################################
# Set the RPATH to make sure executables always work without "make install"
################################################################################
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

################################################################################
# Build type (needs to be handled before project command below)
################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)" FORCE)
endif()

set(APEX_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)" FORCE)

# Always force CMAKE_CONFIGURATION_TYPES to be the same as CMAKE_BUILD_TYPE
# (at least for now, until we figure out how to use multiple build types in
# the same project).
set(CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPE}" CACHE INTERNAL
"Configuration types" FORCE)

# set the project name
project (APEX CXX C)

# include additional macro definitions
include(APEX_Utils)

include(APEX_Distclean)

apex_force_out_of_tree_build("This project requires an out-of-source-tree
build. See README.rst. Clean your CMake cache and CMakeFiles if this message
persists.")

################################################################################
# Compiler detection code
################################################################################

# FIXME: I can't get version detection to work. :(
# so set to ignore for now
# apex_include(GCCVersion)

# The version number.
set (APEX_VERSION_MAJOR 0)
set (APEX_VERSION_MINOR 1)

# add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/apex/apex_config.h.in"
  "${PROJECT_BINARY_DIR}/apex/apex_config.h"
)

# use PIC for shared objects
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") 
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC -g -O0 -Wextra -Waddress -Warray-bounds -Wuninitialized -Wreturn-type")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -O3")

###############################################################################
# reconfigure Boost library information if configuration type has been
# changed since last configure
if(CMAKE_BUILD_TYPE AND NOT (CMAKE_BUILD_TYPE STREQUAL CMAKE_CONFIGURATION_TYPES))
  set(BOOST_VERSION_SEARCHED OFF CACHE INTERNAL "Found Boost version" FORCE)
  set(BOOST_SEARCHED OFF CACHE INTERNAL "Found Boost libraries" FORCE)
endif()

################################################################################
# Boost configuration
################################################################################

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.54.0 REQUIRED COMPONENTS system thread ) 

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    # target_link_libraries(progname ${Boost_LIBRARIES})
endif()

################################################################################
# TAU configuration
################################################################################

# TAU is required.
find_package(TAU REQUIRED)
include_directories(${TAU_INCLUDE_DIRS})
set(LIBS ${LIBS} ${TAU_LIBRARIES})

# RCR is NOT required.
find_package(RCR)
include_directories(${RCR_INCLUDE_DIRS})
set(LIBS ${LIBS} ${RCR_LIBRARIES})

#set (CMAKE_INSTALL_PREFIX ".")
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" ${TAU_LIBRARY_DIR})

# needed for apex.pc.in
SET(prefix ${CMAKE_INSTALL_PREFIX})
SET(exec_prefix "\${prefix}")
SET(libdir "\${exec_prefix}/lib")
SET(bindir "\${exec_prefix}/bin")
SET(includedir "\${prefix}/include")
SET(requirements "")
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/pkgconfig/apex.pc.in
               ${CMAKE_BINARY_DIR}/pkgconfig/apex.pc @ONLY)
INSTALL_FILES(/lib/pkgconfig FILES pkgconfig/apex.pc)

# Recurse into the "Apex" and "Demo" subdirectories. This does not actually 
# cause another cmake executable to run. The same process will walk through 
# the project's entire directory structure. 
add_subdirectory (apex) 
add_subdirectory (ITTNotify) 
add_subdirectory (DemoCpp)
add_subdirectory (DemoC)  
# Add a test program which actually uses the CPU
add_subdirectory (Matmult)  
add_subdirectory (TestITT)
add_subdirectory (TestThreads)

# turn on "make test" support
enable_testing()

# does the application run
add_test (ApexCppRuns DemoCpp/apexCppDemo)
add_test (ApexCRuns DemoC/apexCDemo)

# Run the test program which actually uses the CPU
add_test (MatmultRuns Matmult/matmult)  

# Run the test program which tests the ITT interface
add_test (TestITTRuns TestITT/testITT)  

# Run the test program which tests the threads interface
add_test (TestThreadsRuns TestThreads/testThreads)  
