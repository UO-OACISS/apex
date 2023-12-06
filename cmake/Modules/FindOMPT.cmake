# - Try to find LibOMPT
# Once done this will define
#  OMPT_FOUND - System has OMPT
#  OMPT_INCLUDE_DIRS - The OMPT include directories
#  OMPT_LIBRARIES - The libraries needed to use OMPT
#  OMPT_DEFINITIONS - Compiler switches required for using OMPT

find_package(PkgConfig)

# First, check if the compiler has built-in support for OpenMP 5.0:
INCLUDE(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG(-fopenmp HAVE_OPENMP)
try_compile(APEX_HAVE_OMPT_NATIVE ${CMAKE_CURRENT_BINARY_DIR}/ompt_5.0_test
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/ompt_5.0_test ompt_5.0_test ompt_5.0_test
    CMAKE_FLAGS -DCMAKE_CXX_FLAGS:STRING="${OpenMP_CXX_FLAGS}"
    -DCMAKE_CMAKE_EXE_LINKER_FLAGS:STRING="${OpenMP_CXX_FLAGS}")

if(${APEX_HAVE_OMPT_NATIVE})
    set(OMPT_FOUND TRUE)
    mark_as_advanced(OMPT_INCLUDE_DIR OMPT_LIBRARY)
    #add_custom_target(project_ompt)
    message("Detected compiler has native OpenMP 5.0 Tool support.")

    # Now check for 5.1 support
    try_compile(APEX_HAVE_OMPT_5.1_NATIVE ${CMAKE_CURRENT_BINARY_DIR}/ompt_5.1_test
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/ompt_5.1_test ompt_5.1_test ompt_5.1_test
        CMAKE_FLAGS -DCMAKE_CXX_FLAGS:STRING="${OpenMP_CXX_FLAGS}"
        -DCMAKE_CMAKE_EXE_LINKER_FLAGS:STRING="${OpenMP_CXX_FLAGS}")

    if(${APEX_HAVE_OMPT_5.1_NATIVE})
        message("Detected compiler has native OpenMP 5.1 Tool support.")
        add_definitions(-DAPEX_HAVE_OMPT_5_1)
    else()
        message("Detected compiler does not have native OpenMP 5.1 Tool support.")
    endif()
else(${APEX_HAVE_OMPT_NATIVE})
    set(OMPT_FOUND FALSE)
    message("Detected compiler DOES NOT have native OpenMP 5.0+ Tool support.")
    # Disabled! We don't want to build support any more.
    #include(BuildOMPT)
endif(${APEX_HAVE_OMPT_NATIVE})

if(OMPT_FOUND)
  set(OMPT_LIBRARIES ${OMPT_LIBRARY} )
  set(OMPT_INCLUDE_DIRS ${OMPT_INCLUDE_DIR} )
  set(OMPT_DIR ${OMPT_ROOT})
  add_definitions(-DAPEX_HAVE_OMPT)
endif()

