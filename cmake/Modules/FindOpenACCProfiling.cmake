# - Try to find LibOpenACCProfiling
# Once done this will define
#  OpenACCProfiling_FOUND - System has OpenACCProfiling

find_package(PkgConfig)

# check if the compiler has built-in support for OpenACC profiling
INCLUDE(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG(-fopenmp HAVE_OPENMP)
try_compile(APEX_HAVE_OpenACCProfiling_NATIVE ${CMAKE_CURRENT_BINARY_DIR}/openacc_test
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/openacc_test openacc_test openacc_test
    CMAKE_FLAGS -DCMAKE_CXX_FLAGS:STRING="${OpenACC_CXX_FLAGS}"
    -DCMAKE_CMAKE_EXE_LINKER_FLAGS:STRING="${OpenACC_CXX_FLAGS}")

if(APEX_HAVE_OpenACCProfiling_NATIVE)
    set(OpenACCProfiling_FOUND TRUE)
    message("Detected compiler has OpenACC Profiling support.")
else()
    message("Detected compiler does not have OpenACC Profiling support.")
endif()

