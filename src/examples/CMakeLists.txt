
add_subdirectory (DemoCpp)
add_subdirectory (DemoC)  
# Add a test program which actually uses the CPU
add_subdirectory (Matmult)  
if(MPI_CXX_FOUND)
  add_subdirectory (MPITest)  
  add_subdirectory (MPIImbalancePolicy)  
  add_subdirectory (LuleshMPI)  
  if(OPENMP_FOUND)
    add_subdirectory (MPIGlobalTest)  
    add_subdirectory (LuleshMPIOpenMP)  
  endif()
endif()
add_subdirectory (TestITT)
add_subdirectory (TestThreads)
add_subdirectory (CountCalls)
add_subdirectory (Overhead)
add_subdirectory (PolicyUnitTest)
add_subdirectory (PolicyEngineExample)
add_subdirectory (PolicyEngineCppExample)
add_subdirectory (TestReset)
add_subdirectory (TestGetPower)
add_subdirectory (Throttling)
add_subdirectory (ThrottlingCpp)
if (ACTIVEHARMONY_FOUND)
  add_subdirectory (ThrottlingActiveHarmony)
endif()
if(OPENMP_FOUND)
  add_subdirectory (OpenMP_1D_stencil)
  add_subdirectory (LuleshOpenMP)
  if(OMPT_FOUND)
    add_subdirectory (TestOpenMP)
  endif()
endif()
if(USE_UDP_CLIENT)
add_subdirectory (UdpSink)
endif(USE_UDP_CLIENT)
add_subdirectory (CustomTuning)

# does the application run
add_test (TestDemoC DemoC/apexCDemo)
set_tests_properties(TestDemoC PROPERTIES PASS_REGULAR_EXPRESSION "APEX Version : ")
add_test (TestDemoCpp DemoCpp/apexCppDemo)
set_tests_properties(TestDemoCpp PROPERTIES PASS_REGULAR_EXPRESSION "APEX Version : ")

# Run the test program which actually uses the CPU
add_test (TestMatmult Matmult/matmult)  
set_tests_properties(TestMatmult PROPERTIES PASS_REGULAR_EXPRESSION "Done.")

# Run the test program which tests the ITT interface
add_test (TestTestITT TestITT/testITT)  
set_tests_properties(TestTestITT PROPERTIES PASS_REGULAR_EXPRESSION "ITT demo.")

# Run the test program which tests the threads interface
add_test (TestTestThreads TestThreads/testThreads)  
set_tests_properties(TestTestThreads PROPERTIES PASS_REGULAR_EXPRESSION "The ID of this thread is:")

# Run the test program which tests the yielding
add_test (TestCountCalls CountCalls/count_calls)  
set_tests_properties(TestCountCalls PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")

# Run the test program which tests the overhead in the threads interface
add_test (TestOverhead Overhead/testOverhead)  
set_tests_properties(TestOverhead PROPERTIES PASS_REGULAR_EXPRESSION "Average overhead per timer")

# TEst the policy engine support
add_test (TestPolicyUnitTest PolicyUnitTest/policyUnitTest)  
set_tests_properties(TestPolicyUnitTest PROPERTIES ENVIRONMENT "APEX_POLICY=1")
set_tests_properties(TestPolicyUnitTest PROPERTIES FAIL_REGULAR_EXPRESSION "Warning: registered policy function failed!")

# TEst the policy engine support
add_test (TestPolicyEngineExample PolicyEngineExample/testPolicyEngine)  
set_tests_properties(TestPolicyEngineExample PROPERTIES ENVIRONMENT "APEX_POLICY=1")
set_tests_properties(TestPolicyEngineExample PROPERTIES FAIL_REGULAR_EXPRESSION "Warning: registered policy function failed!")

add_test (TestPolicyEngineCppExample PolicyEngineCppExample/testCppPolicyEngine)  
set_tests_properties(TestPolicyEngineCppExample PROPERTIES ENVIRONMENT "APEX_POLICY=1")
set_tests_properties(TestPolicyEngineCppExample PROPERTIES FAIL_REGULAR_EXPRESSION "Warning: registered policy function failed!")

# Test the resetting of counters and timers
add_test (TestReset TestReset/testReset)
set_tests_properties(TestReset PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")

# Test the power/energy support
add_test (TestGetPower TestGetPower/testpower)
set_tests_properties(TestGetPower PROPERTIES PASS_REGULAR_EXPRESSION "Power at end:")

# Test the throttling support
add_test (TestThrottling Throttling/testThrottling)  
set_tests_properties(TestThrottling PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")
set_property(TEST TestThrottling PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST TestThrottling APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
set_property(TEST TestThrottling APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_ENERGY=1")

add_test (TestThrottlingCpp ThrottlingCpp/testThrottlingCpp)  
set_tests_properties(TestThrottlingCpp PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")
set_property(TEST TestThrottlingCpp PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST TestThrottlingCpp APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
set_property(TEST TestThrottling APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_ENERGY=1")

if (ACTIVEHARMONY_FOUND)
add_test (TestThrottlingActiveHarmony ThrottlingActiveHarmony/testThrottlingActiveHarmony)  
set_tests_properties(TestThrottlingActiveHarmony PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")
set_property(TEST TestThrottlingActiveHarmony PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST TestThrottlingActiveHarmony APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_ENERGY=1")
set_property(TEST TestThrottlingActiveHarmony APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
set_property(TEST TestThrottlingActiveHarmony APPEND PROPERTY ENVIRONMENT "HARMONY_HOME=${ACTIVEHARMONY_ROOT}")
endif()

if(MPI_CXX_FOUND)
# Run the test program which tests the mpi interface
  add_test (TestMPITest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} MPITest/mpi_test ${MPIEXEC_POSTFLAGS})  
  set_tests_properties(TestMPITest PROPERTIES ENVIRONMENT "APEX_POLICY=1")
  add_test (TestMPIGlobalTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} MPIGlobalTest/mpi_global_test ${MPIEXEC_POSTFLAGS})  
  set_tests_properties(TestMPIGlobalTest PROPERTIES ENVIRONMENT "APEX_POLICY=1")
  set_property(TEST TestMPIGlobalTest APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_ENERGY=1")
  set_property(TEST TestMPIGlobalTest APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
  add_test (TestMPIImbalancePolicy ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} MPIImbalancePolicy/mpi_imbalance_test ${MPIEXEC_POSTFLAGS})  
  set_tests_properties(TestMPIImbalancePolicy PROPERTIES ENVIRONMENT "APEX_POLICY=1")
#  add_test (TestLuleshMPI ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8
#    ${MPIEXEC_PREFLAGS} LuleshMPI/lulesh_MPI_2.0 -s 15 ${MPIEXEC_POSTFLAGS})  
#  if(OPENMP_FOUND)
#    add_test (TestLuleshMPIOpenMP ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8
#      ${MPIEXEC_PREFLAGS} LuleshMPIOpenMP/lulesh_MPI_OpenMP_2.0
#      -s 15 ${MPIEXEC_POSTFLAGS})  
#  endif()
endif()

if(OPENMP_FOUND)
    add_test (TestOpenMPStencil OpenMP_1D_stencil/1d_stencil)
    if (ACTIVEHARMONY_FOUND)
        set_tests_properties(TestOpenMPStencil PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")
        set_property(TEST TestOpenMPStencil PROPERTY ENVIRONMENT "APEX_POLICY=1")
        set_property(TEST TestOpenMPStencil APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_ENERGY=1")
        set_property(TEST TestOpenMPStencil APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
        set_property(TEST TestOpenMPStencil APPEND PROPERTY ENVIRONMENT "HARMONY_HOME=${ACTIVEHARMONY_ROOT}")
    endif()
#  add_test (TestLuleshOpenMP LuleshOpenMP/lulesh_OpenMP_2.0 -s 15)
  if(OMPT_FOUND)
      add_test (TestOpenMPSharing TestOpenMP/openmp_test)
      if (PROCESSOR_COUNT)
        set_property (TEST TestOpenMPSharing
           APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PROCESSOR_COUNT}")
      endif (PROCESSOR_COUNT)
  endif()
  if (PROCESSOR_COUNT)
    set_property (TEST TestOpenMPStencil
          APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PROCESSOR_COUNT}")
#    if(MPI_CXX_FOUND)
#      set_property (TEST TestLuleshMPIOpenMP
#            APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PROCESSOR_COUNT}")
#    endif(MPI_CXX_FOUND)
  endif(PROCESSOR_COUNT)
endif()

if(USE_UDP_CLIENT)
    # Run the test program which tests the UDP network reporting
    add_test (TestUdpSink UdpSink/test_script.sh)  
    set_tests_properties(TestUdpSink PROPERTIES PASS_REGULAR_EXPRESSION "The ID of this thread is:")
endif(USE_UDP_CLIENT)

if (PAPI_FOUND)
    set_property (TEST
        TestDemoC
        TestDemoCpp
        TestMatmult
        TestTestITT
        TestTestThreads
        TestCountCalls
        TestOverhead
        TestPolicyUnitTest
        TestPolicyEngineExample
        TestPolicyEngineCppExample
        TestReset
        TestGetPower
        TestThrottling
        TestThrottlingCpp
        APPEND PROPERTY ENVIRONMENT 
         "APEX_PAPI_METRICS=PAPI_TOT_CYC PAPI_TOT_INS PAPI_LD_INS")
endif (PAPI_FOUND)

