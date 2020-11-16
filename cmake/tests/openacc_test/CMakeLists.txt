project(openacc_test)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${OpenACC_CXX_FLAGS}")
add_executable(openacc_test openacc_test.cpp) 