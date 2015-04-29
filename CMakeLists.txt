# The name of our project is "APEX". CMakeLists files in this project can
# refer to the root source directory of the project as ${APEX_SOURCE_DIR} and
# to the root binary directory of the project as ${APEX_BINARY_DIR}.
cmake_minimum_required (VERSION 2.8.12 FATAL_ERROR)

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

if(APEX_THROTTLE)
    # add_definitions(-DAPEX_THROTTLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPEX_THROTTLE")
endif(APEX_THROTTLE)

# include additional macro definitions
include(APEX_Utils)

include(APEX_Distclean)

apex_force_out_of_tree_build("This project requires an out-of-source-tree
build. See README.rst. Clean your CMake cache and CMakeFiles if this message
persists.")

################################################################################
# Get the GIT version of the code
################################################################################

# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

#add_definitions("-DGIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
#add_definitions("-DGIT_BRANCH=${GIT_BRANCH}")

################################################################################
# Hostname detection code
################################################################################

site_name(HOSTNAME)
string(LENGTH ${HOSTNAME} HOSTNAME_LENGTH)
if(${HOSTNAME_LENGTH} GREATER 5)
	string(SUBSTRING ${HOSTNAME} 0 6 HOST_BASENAME)
else()
	set (HOST_BASENAME ${HOSTNAME})
endif()

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
  "${PROJECT_SOURCE_DIR}/src/apex/apex_config.h.in"
  "${PROJECT_BINARY_DIR}/src/apex/apex_config.h"
)

# use PIC for shared objects
#if (BUILD_SHARED_LIBS)
set(CMAKE_POSITION_INDEPENDENT_CODE True)
set(POSITION_INDEPENDENT_CODE True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
#endif (BUILD_SHARED_LIBS)

#set debugging info
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -g -O0 -Wextra -Waddress -Warray-bounds -Wuninitialized -Wreturn-type -Wall -fno-omit-frame-pointer -DAPEX_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -O3")

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

if (APPLE)
     set(Boost_USE_STATIC_LIBS ON)
else()
    set(Boost_USE_STATIC_LIBS OFF)
endif()
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.54.0 REQUIRED COMPONENTS system thread timer chrono regex program_options)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    # target_link_libraries(progname ${Boost_LIBRARIES})
endif()

################################################################################
# JSON configuration
################################################################################

include(GitExternal)
git_external(rapidjson
    https://github.com/miloyip/rapidjson.git
    master
    VERBOSE)

################################################################################
# MPI configuration
################################################################################

if(USE_MPI)
	find_package(MPI)
endif()

################################################################################
# TAU configuration
################################################################################

# TAU is NOT required.
if (DEFINED TAU_ROOT)
find_package(TAU)
if (TAU_FOUND)
	include_directories(${TAU_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${TAU_LIBRARIES})
	set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${TAU_LIBRARY_DIR})
endif()
endif()

################################################################################
# TAU configuration
################################################################################

# RCR is NOT required.
if (DEFINED RCR_ROOT)
    find_package(RCR)
endif()

if (RCR_FOUND)
    include_directories(${RCR_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${RCR_LIBRARIES})
    set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${RCR_LIBRARY_DIR})
else()
	#if(EXISTS "/sys/cray/pm_counter/power")
	IF("${HOST_BASENAME}" STREQUAL "edison")
		set (APEX_HAVE_CRAY_POWER TRUE)
		add_definitions(-DAPEX_HAVE_CRAY_POWER)
	endif()
endif()

find_package(OpenMP)
 
if (OPENMP_FOUND)
  if (USE_OMPT)
      find_package(OMPT)
      if (OMPT_FOUND)
          include_directories(${OMPT_INCLUDE_DIRS})
          set(LIBS ${LIBS} ${OMPT_LIBRARIES})
          set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${OMPT_LIBRARY_DIR})
      endif()
  endif()
endif()

if(EXISTS "/proc/stat")
	set (APEX_HAVE_PROC TRUE)
	add_definitions(-DAPEX_HAVE_PROC)
endif()

################################################################################
# TCMalloc configuration
################################################################################

find_package(Tcmalloc)
if (Tcmalloc_FOUND)
	include_directories(${Tcmalloc_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${Tcmalloc_LIBRARIES})
	set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${Tcmalloc_LIBRARY_DIR})
endif()

################################################################################
# ActiveHarmony configuration
################################################################################

if(USE_ACTIVEHARMONY)
    find_package(ActiveHarmony)
    if (ACTIVEHARMONY_FOUND)
        include_directories(${ACTIVEHARMONY_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${ACTIVEHARMONY_LIBRARIES})
        set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${ACTIVEHARMONY_LIBRARY_DIR})
    endif()
endif()

################################################################################
# PAPI configuration
################################################################################

if(DEFINED $PAPI_ROOT)
    set (USE_PAPI TRUE)
endif()

if(USE_PAPI)
	find_package(PAPI)
	if (PAPI_FOUND)
		include_directories(${PAPI_INCLUDE_DIRS})
		set(LIBS ${LIBS} ${PAPI_LIBRARIES})
		set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${PAPI_LIBRARY_DIR})
	endif()
endif()

################################################################################
# Binutils configuration
################################################################################

if(USE_BINUTILS)
	find_package(BFD)
	if (BFD_FOUND)
		include_directories(${BFD_INCLUDE_DIRS})
		set(LIBS ${LIBS} ${BFD_LIBRARIES})
		set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${BFD_LIBRARY_DIR})
	endif()

	find_package(Demangle)
	if (DEMANGLE_FOUND)
		include_directories(${DEMANGLE_INCLUDE_DIRS})
		set(LIBS ${LIBS} ${DEMANGLE_LIBRARIES})
		set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${DEMANGLE_LIBRARY_DIR})
	else()
  	unset(DEMANGLE_LIBRARY)
  	unset(DEMANGLE_LIBRARIES)
  	unset(DEMANGLE_DIR)
	endif()
	find_package(ZLIB)
	if (ZLIB_FOUND)
		set(LIBS ${LIBS} ${ZLIB_LIBRARIES})
		set (CMAKE_INSTALL_RPATH ${ZLIB_LIBRARY_DIR})
	endif()
endif(USE_BINUTILS)

set (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} "${CMAKE_INSTALL_PREFIX}/lib")

# needed for apex.pc.in
SET(prefix ${CMAKE_INSTALL_PREFIX})
SET(exec_prefix "\${prefix}")
SET(libdir "\${exec_prefix}/lib")
SET(bindir "\${exec_prefix}/bin")
SET(includedir "\${prefix}/include")
string(REPLACE ";" " " TMP_LIBS "${LIBS}")
string(REPLACE ";" " " TMP_Boost "${Boost_LIBRARIES}")
SET(extralibs "${TMP_LIBS} ${TMP_Boost} ${CMAKE_EXE_LINKER_FLAGS}")
SET(extraincludes "-I${RCR_INCLUDE_DIR}")
SET(requirements "")
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/pkgconfig/apex.pc.in
               ${CMAKE_BINARY_DIR}/pkgconfig/apex.pc @ONLY)
INSTALL_FILES(/lib/pkgconfig FILES pkgconfig/apex.pc)

# Recurse into the "Apex" and "Demo" subdirectories. This does not actually 
# cause another cmake executable to run. The same process will walk through 
# the project's entire directory structure. 
add_subdirectory (src/apex) 
add_subdirectory (src/ITTNotify) 
add_subdirectory (src/examples/DemoCpp)
add_subdirectory (src/examples/DemoC)  
# Add a test program which actually uses the CPU
add_subdirectory (src/examples/Matmult)  
if(MPI_CXX_FOUND)
  add_subdirectory (src/comm) 
  add_subdirectory (src/comm/mpi) 
  add_subdirectory (src/examples/MPITest)  
  add_subdirectory (src/examples/MPIGlobalTest)  
  add_subdirectory (src/examples/MPIImbalancePolicy)  
endif()
add_subdirectory (src/examples/TestITT)
add_subdirectory (src/examples/TestThreads)
add_subdirectory (src/examples/CountCalls)
add_subdirectory (src/examples/Overhead)
add_subdirectory (src/examples/PolicyEngineExample)
add_subdirectory (src/examples/PolicyEngineCppExample)
add_subdirectory (src/examples/TestReset)
add_subdirectory (src/examples/TestGetPower)
add_subdirectory (src/examples/Throttling)
add_subdirectory (src/examples/ThrottlingCpp)
if (ACTIVEHARMONY_FOUND)
  add_subdirectory (src/examples/ThrottlingActiveHarmony)
endif()
if(OPENMP_FOUND)
  add_subdirectory (src/examples/OpenMP_1D_stencil)
  if(OMPT_FOUND)
    add_subdirectory (src/examples/TestOpenMP)
  endif()
endif()
add_subdirectory (src/udp_server)
add_subdirectory (src/examples/Beacon)
add_subdirectory (doc) 

# turn on "make test" support
enable_testing()

# does the application run
add_test (TestDemoC src/examples/DemoC/apexCDemo)
set_tests_properties(TestDemoC PROPERTIES PASS_REGULAR_EXPRESSION "APEX Version : ")
add_test (TestDemoCpp src/examples/DemoCpp/apexCppDemo)
set_tests_properties(TestDemoCpp PROPERTIES PASS_REGULAR_EXPRESSION "APEX Version : ")

# Run the test program which actually uses the CPU
add_test (TestMatmult src/examples/Matmult/matmult)  
set_tests_properties(TestMatmult PROPERTIES PASS_REGULAR_EXPRESSION "Done.")

# Run the test program which tests the ITT interface
add_test (TestTestITT src/examples/TestITT/testITT)  
set_tests_properties(TestTestITT PROPERTIES PASS_REGULAR_EXPRESSION "ITT demo.")

# Run the test program which tests the threads interface
add_test (TestTestThreads src/examples/TestThreads/testThreads)  
set_tests_properties(TestTestThreads PROPERTIES PASS_REGULAR_EXPRESSION "The ID of this thread is:")

# Run the test program which tests the yielding
add_test (TestCountCalls src/examples/CountCalls/count_calls)  
set_tests_properties(TestCountCalls PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")

# Run the test program which tests the overhead in the threads interface
add_test (TestOverhead src/examples/Overhead/testOverhead)  
set_tests_properties(TestOverhead PROPERTIES PASS_REGULAR_EXPRESSION "The ID of this thread is:")

# TEst the policy engine support
add_test (TestPolicyEngineExample src/examples/PolicyEngineExample/testPolicyEngine)  
set_tests_properties(TestPolicyEngineExample PROPERTIES ENVIRONMENT "APEX_POLICY=1")
set_tests_properties(TestPolicyEngineExample PROPERTIES FAIL_REGULAR_EXPRESSION "Warning: registered policy function failed!")

add_test (TestPolicyEngineCppExample src/examples/PolicyEngineCppExample/testCppPolicyEngine)  
set_tests_properties(TestPolicyEngineCppExample PROPERTIES ENVIRONMENT "APEX_POLICY=1")
set_tests_properties(TestPolicyEngineCppExample PROPERTIES FAIL_REGULAR_EXPRESSION "Warning: registered policy function failed!")

# Test the resetting of counters and timers
add_test (TestReset src/examples/TestReset/testReset)
set_tests_properties(TestReset PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")

# Test the power/energy support
add_test (TestGetPower src/examples/TestGetPower/testpower)
set_tests_properties(TestGetPower PROPERTIES PASS_REGULAR_EXPRESSION "Power at end:")

# Test the throttling support
add_test (TestThrottling src/examples/Throttling/testThrottling)  
set_tests_properties(TestThrottling PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")
set_property(TEST TestThrottling PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST TestThrottling APPEND PROPERTY ENVIRONMENT "APEX_THROTTLING=1")

add_test (TestThrottlingCpp src/examples/ThrottlingCpp/testThrottlingCpp)  
set_tests_properties(TestThrottlingCpp PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")
set_property(TEST TestThrottlingCpp PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST TestThrottlingCpp APPEND PROPERTY ENVIRONMENT "APEX_THROTTLING=1")

if (ACTIVEHARMONY_FOUND)
add_test (TestThrottlingActiveHarmony src/examples/ThrottlingActiveHarmony/testThrottlingActiveHarmony)  
set_tests_properties(TestThrottlingActiveHarmony PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")
set_property(TEST TestThrottlingActiveHarmony PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST TestThrottlingActiveHarmony APPEND PROPERTY ENVIRONMENT "APEX_THROTTLING=1")
set_property(TEST TestThrottlingActiveHarmony APPEND PROPERTY ENVIRONMENT "HARMONY_HOME=${ACTIVEHARMONY_ROOT}")
endif()

if(MPI_CXX_FOUND)
# Run the test program which tests the mpi interface
  add_test (TestMPITest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} src/examples/MPITest/mpi_test ${MPIEXEC_POSTFLAGS})  
  set_tests_properties(TestMPITest PROPERTIES ENVIRONMENT "APEX_POLICY=1")
  add_test (TestMPIGlobalTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} src/examples/MPIGlobalTest/mpi_global_test ${MPIEXEC_POSTFLAGS})  
  set_tests_properties(TestMPIGlobalTest PROPERTIES ENVIRONMENT "APEX_POLICY=1")
  add_test (TestMPIImbalancePolicy ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} src/examples/MPIImbalancePolicy/mpi_imbalance_test ${MPIEXEC_POSTFLAGS})  
  set_tests_properties(TestMPIImbalancePolicy PROPERTIES ENVIRONMENT "APEX_POLICY=1")
endif()

if(OPENMP_FOUND)
    add_test (TestOpenMPStencil src/examples/OpenMP_1D_stencil/1d_stencil)
    if (ACTIVEHARMONY_FOUND)
        set_tests_properties(TestOpenMPStencil PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")
        set_property(TEST TestOpenMPStencil PROPERTY ENVIRONMENT "APEX_POLICY=1")
        set_property(TEST TestOpenMPStencil APPEND PROPERTY ENVIRONMENT "APEX_THROTTLING=1")
        set_property(TEST TestOpenMPStencil APPEND PROPERTY ENVIRONMENT "HARMONY_HOME=${ACTIVEHARMONY_ROOT}")
    endif()
  if(OMPT_FOUND)
	  add_test (TestOpenMP src/examples/TestOpenMP/openmp_test)
  endif()
endif()

# Run the test program which tests the beacon network reporting
add_test (TestBeacon src/examples/Beacon/test_script.sh)  
set_tests_properties(TestBeacon PROPERTIES PASS_REGULAR_EXPRESSION "The ID of this thread is:")

