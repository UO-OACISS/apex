# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

apex_include(Option)

set(APEX_GCCVERSION_LOADED TRUE)

set(source_dir "${PROJECT_SOURCE_DIR}/cmake/tests")

if(NOT MSVC)
  include(APEX_Include)

  apex_include(Compile GetIncludeDirectory)

  apex_get_include_directory(include_dir)

  if(apex_SOURCE_DIR)
    set(source_dir "${apex_SOURCE_DIR}/cmake/tests")
  elseif(APEX_ROOT)
    set(source_dir "${APEX_ROOT}/share/apex/cmake/tests")
  elseif($ENV{APEX_ROOT})
    set(source_dir "$ENV{APEX_ROOT}/share/apex/cmake/tests")
  endif()

  file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests)

  apex_compile(gcc_version SOURCE ${source_dir}/gcc_version.cpp
    LANGUAGE CXX
    OUTPUT ${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/gcc_version
    FLAGS -I${BOOST_INCLUDE_DIR} ${include_dir})

  if("${gcc_version_RESULT}" STREQUAL "0")
    if(NOT GCC_VERSION)
      execute_process(
        COMMAND "${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/gcc_version"
        OUTPUT_VARIABLE GCC_VERSION)
    endif()

    if("${GCC_VERSION}" STREQUAL "")
      set(GCC_VERSION "000000" CACHE INTERNAL "" FORCE)
      set(GCC_VERSION_NUM "000000" CACHE INTERNAL "" FORCE)
      set(GCC_MAJOR_VERSION "00" CACHE INTERNAL "" FORCE)
      set(GCC_MINOR_VERSION "00" CACHE INTERNAL "" FORCE)
      set(GCC_PATCH_VERSION "00" CACHE INTERNAL "" FORCE)
      set(GCC_VERSION_STR "unknown" CACHE INTERNAL "" FORCE)
    else()
      math(EXPR GCC_MAJOR_VERSION "${GCC_VERSION} / 10000")
      math(EXPR GCC_MINOR_VERSION "${GCC_VERSION} / 100 % 100")
      math(EXPR GCC_PATCH_VERSION "${GCC_VERSION} % 100")

      set(GCC_VERSION "${GCC_VERSION}" CACHE INTERNAL "" FORCE)
      set(GCC_VERSION_NUM "${GCC_VERSION}" CACHE INTERNAL "" FORCE)
      set(GCC_MAJOR_VERSION "${GCC_MAJOR_VERSION}" CACHE INTERNAL "" FORCE)
      set(GCC_MINOR_VERSION "${GCC_MINOR_VERSION}" CACHE INTERNAL "" FORCE)
      set(GCC_PATCH_VERSION "${GCC_PATCH_VERSION}" CACHE INTERNAL "" FORCE)
      set(GCC_VERSION_STR
        "${GCC_MAJOR_VERSION}.${GCC_MINOR_VERSION}.${GCC_PATCH_VERSION}"
        CACHE INTERNAL "" FORCE)
    endif()
  else()
    set(GCC_VERSION "000000" CACHE INTERNAL "" FORCE)
    set(GCC_VERSION_NUM "000000" CACHE INTERNAL "" FORCE)
    set(GCC_MAJOR_VERSION "00" CACHE INTERNAL "" FORCE)
    set(GCC_MINOR_VERSION "00" CACHE INTERNAL "" FORCE)
    set(GCC_PATCH_VERSION "00" CACHE INTERNAL "" FORCE)
    set(GCC_VERSION_STR "unknown" CACHE INTERNAL "" FORCE)
  endif()
endif()

################################################################################
# Compiler detection code
################################################################################

# C++

if(GCC_VERSION AND NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
               AND NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
  apex_info("gcc_config" "Compiler reports compatibility with GCC version
${GCC_VERSION_STR}")

  apex_option(APEX_IGNORE_GCC_VERSION BOOL
    "Ignore version reported by gcc (default: OFF)." OFF ADVANCED)

  if(APEX_IGNORE_GCC_VERSION)
    apex_warn("gcc_config" "GCC 4.4.5 or higher is required. Building APEX will
proceed but may fail.")
  elseif(040405 GREATER ${GCC_VERSION})
    apex_error("gcc_config" "GCC 4.4.5 or higher is required. Specify
APEX_IGNORE_GCC_VERSION=ON to overwrite this error.")
  endif()
elseif(MSVC)
  if(NOT (MSVC10 OR MSVC11))
    apex_error("msvc_config" "MSVC x64 2010 or higher is required.")
  elseif(NOT CMAKE_CL_64)
    apex_warn("msvc_config" "MSVC (32Bit) will compile but will fail running
larger applications because of limitations in the Windows OS.")
  endif()
endif()


