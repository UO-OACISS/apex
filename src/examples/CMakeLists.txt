
#add_subdirectory (PthreadWrapper)
add_subdirectory (DemoCpp)
add_subdirectory (DemoC)
# Add a test program which actually uses the CPU
add_subdirectory (Matmult)
if(MPI_CXX_FOUND)
  add_subdirectory (MPITest)
  add_subdirectory (LuleshMPI)
  if(OPENMP_FOUND)
    add_subdirectory (MPIImbalancePolicy)
    #add_subdirectory (MPIGlobalTest)
    add_subdirectory (LuleshMPIOpenMP)
    add_subdirectory (MPIAutoBalance)
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
  add_subdirectory (LuleshOpenMP)
  add_subdirectory (OpenMPOverhead)
  if(OMPT_FOUND)
    add_subdirectory (TestOpenMP)
    if(PROCESSOR_COUNT)
      if(PROCESSOR_COUNT LESS 33)
  		add_subdirectory (OpenMP_1D_stencil)
	  endif()
    endif()
    if (NOT BUILD_STATIC_EXECUTABLES AND APEX_WITH_PLUGINS)
  	  add_subdirectory (OpenMP_Policy)
    endif()
  endif()
endif()
add_subdirectory (CustomTuning)
add_subdirectory (TuningRequest)
add_subdirectory (EventFilter)

# does the application run
add_test (ExampleDemoC DemoC/apexCDemo)
set_tests_properties(ExampleDemoC PROPERTIES PASS_REGULAR_EXPRESSION "APEX Version : ")
add_test (ExampleDemoCpp DemoCpp/apexCppDemo)
set_tests_properties(ExampleDemoCpp PROPERTIES PASS_REGULAR_EXPRESSION "APEX Version : ")

# Run the test program which actually uses the CPU
add_test (ExampleMatmult Matmult/matmult)
set_tests_properties(ExampleMatmult PROPERTIES PASS_REGULAR_EXPRESSION "Done.")

# Run the test program which tests the ITT interface
add_test (ExampleTestITT TestITT/testITT)
set_tests_properties(ExampleTestITT PROPERTIES PASS_REGULAR_EXPRESSION "ITT demo.")

# Run the test program which tests the threads interface
add_test (ExampleTestThreads TestThreads/testThreads)
set_tests_properties(ExampleTestThreads PROPERTIES PASS_REGULAR_EXPRESSION "The ID of this thread is:")

# Run the test program which tests the yielding
add_test (ExampleCountCalls CountCalls/count_calls)
set_tests_properties(ExampleCountCalls PROPERTIES PASS_REGULAR_EXPRESSION
"Test passed.")

# Run the test program which tests the overhead in the threads interface
add_test (ExampleOverhead Overhead/testOverhead)
set_tests_properties(ExampleOverhead PROPERTIES PASS_REGULAR_EXPRESSION "Estimated overhead per timer")

# TEst the policy engine support
add_test (ExamplePolicyUnitTest PolicyUnitTest/policyUnitTest)
set_tests_properties(ExamplePolicyUnitTest PROPERTIES ENVIRONMENT "APEX_POLICY=1")
set_tests_properties(ExamplePolicyUnitTest PROPERTIES FAIL_REGULAR_EXPRESSION "Warning: registered policy function failed!")

# TEst the policy engine support
add_test (ExamplePolicyEngineExample PolicyEngineExample/testPolicyEngine)
set_tests_properties(ExamplePolicyEngineExample PROPERTIES ENVIRONMENT "APEX_POLICY=1")
set_tests_properties(ExamplePolicyEngineExample PROPERTIES FAIL_REGULAR_EXPRESSION "Warning: registered policy function failed!")

add_test (ExamplePolicyEngineCppExample PolicyEngineCppExample/testCppPolicyEngine)
set_tests_properties(ExamplePolicyEngineCppExample PROPERTIES ENVIRONMENT "APEX_POLICY=1")
set_tests_properties(ExamplePolicyEngineCppExample PROPERTIES FAIL_REGULAR_EXPRESSION "Warning: registered policy function failed!")

# Test the resetting of counters and timers
add_test (ExampleReset TestReset/testReset)
set_tests_properties(ExampleReset PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")

# Test the power/energy support
add_test (ExampleGetPower TestGetPower/testpower)
set_tests_properties(ExampleGetPower PROPERTIES PASS_REGULAR_EXPRESSION "Power at end:")

# Test the throttling support
add_test (ExampleThrottling Throttling/testThrottling)
set_tests_properties(ExampleThrottling PROPERTIES PASS_REGULAR_EXPRESSION
"Test passed.")
set_property(TEST ExampleThrottling PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST ExampleThrottling APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
set_property(TEST ExampleThrottling APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_ENERGY=1")

add_test (ExampleThrottlingCpp ThrottlingCpp/testThrottlingCpp)
set_tests_properties(ExampleThrottlingCpp PROPERTIES PASS_REGULAR_EXPRESSION
"Test passed.")
set_property(TEST ExampleThrottlingCpp PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST ExampleThrottlingCpp APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
set_property(TEST ExampleThrottling APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_ENERGY=1")

if (ACTIVEHARMONY_FOUND)
add_test (ExampleThrottlingActiveHarmony ThrottlingActiveHarmony/testThrottlingActiveHarmony)
set_tests_properties(ExampleThrottlingActiveHarmony PROPERTIES
PASS_REGULAR_EXPRESSION "Test passed.")
set_property(TEST ExampleThrottlingActiveHarmony PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST ExampleThrottlingActiveHarmony APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_ENERGY=1")
set_property(TEST ExampleThrottlingActiveHarmony APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
set_property(TEST ExampleThrottlingActiveHarmony APPEND PROPERTY ENVIRONMENT "HARMONY_HOME=${ACTIVEHARMONY_ROOT}")

add_test (ExampleCustomTuning CustomTuning/custom_tuning)
set_tests_properties(ExampleCustomTuning PROPERTIES
PASS_REGULAR_EXPRESSION "Test passed.")
set_property(TEST ExampleCustomTuning PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST ExampleCustomTuning APPEND PROPERTY ENVIRONMENT "HARMONY_HOME=${ACTIVEHARMONY_ROOT}")

add_test (ExampleTuningRequest TuningRequest/tuning_request)
set_tests_properties(ExampleTuningRequest PROPERTIES
PASS_REGULAR_EXPRESSION "Test passed.")
set_property(TEST ExampleTuningRequest PROPERTY ENVIRONMENT "APEX_POLICY=1")
set_property(TEST ExampleTuningRequest APPEND PROPERTY ENVIRONMENT "HARMONY_HOME=${ACTIVEHARMONY_ROOT}")

endif()

if(MPI_CXX_FOUND)
# Run the test program which tests the mpi interface
  add_test (ExampleMPITest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} MPITest/mpi_test ${MPIEXEC_POSTFLAGS})
  set_tests_properties(ExampleMPITest PROPERTIES ENVIRONMENT "APEX_POLICY=1")
  string(TIMESTAMP VERSION "%Y-%m-%d_%H.%M.%S")
  set_property(TEST ExampleMPITest APPEND PROPERTY ENVIRONMENT "APEX_OTF2_ARCHIVE_PATH=OTF2_archive_mpi_test_${VERSION}")
  #add_test (ExampleMPIGlobalTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
  #${MPIEXEC_PREFLAGS} MPIGlobalTest/mpi_global_test ${MPIEXEC_POSTFLAGS})
  #set_tests_properties(ExampleMPIGlobalTest PROPERTIES ENVIRONMENT "APEX_POLICY=1")
  #set_property(TEST ExampleMPIGlobalTest APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_ENERGY=1")
  #set_property(TEST ExampleMPIGlobalTest APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
  add_test (ExampleMPIImbalancePolicy ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} MPIImbalancePolicy/mpi_imbalance_test ${MPIEXEC_POSTFLAGS})
  set_tests_properties(ExampleMPIImbalancePolicy PROPERTIES ENVIRONMENT "APEX_POLICY=1")
  string(TIMESTAMP VERSION "%Y-%m-%d_%H.%M.%S")
  set_property(TEST ExampleMPIImbalancePolicy APPEND PROPERTY ENVIRONMENT "APEX_OTF2_ARCHIVE_PATH=OTF2_archive_mpi_imbalance_test_${VERSION}")
#  add_test (ExampleLuleshMPI ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8
#    ${MPIEXEC_PREFLAGS} LuleshMPI/lulesh_MPI_2.0 -s 15 ${MPIEXEC_POSTFLAGS})
  if(OPENMP_FOUND)
#    add_test (ExampleLuleshMPIOpenMP ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8
#      ${MPIEXEC_PREFLAGS} LuleshMPIOpenMP/lulesh_MPI_OpenMP_2.0
#      -s 15 ${MPIEXEC_POSTFLAGS})
    add_test (ExampleMPIAutoBalance ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
      ${MPIEXEC_PREFLAGS} MPIAutoBalance/mpi_auto_balance ${MPIEXEC_POSTFLAGS})
      if (PROCESSOR_COUNT)
        set_tests_properties(ExampleMPIAutoBalance PROPERTIES
		PASS_REGULAR_EXPRESSION "Test passed.")
        string(TIMESTAMP VERSION "%Y-%m-%d_%H.%M.%S")
        set_property(TEST ExampleMPIAutoBalance APPEND PROPERTY ENVIRONMENT "APEX_OTF2_ARCHIVE_PATH=OTF2_archive_mpi_auto_balance_${VERSION}")
        set_property (TEST ExampleMPIAutoBalance APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PROCESSOR_COUNT}")
       set_property(TEST ExampleMPIAutoBalance APPEND PROPERTY ENVIRONMENT "APEX_POLICY=1")
        set_property(TEST ExampleMPIAutoBalance APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
    endif()
  endif()
endif()

if(OPENMP_FOUND)
#  add_test (ExampleLuleshOpenMP LuleshOpenMP/lulesh_OpenMP_2.0 -s 15)
  if(OMPT_FOUND)
      add_test (ExampleOpenMPSharing TestOpenMP/openmp_test)
      if (PROCESSOR_COUNT)
        set_property (TEST ExampleOpenMPSharing
           APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PROCESSOR_COUNT}")
      endif (PROCESSOR_COUNT)
    if (PROCESSOR_COUNT)
      if(PROCESSOR_COUNT LESS 16)
        add_test (ExampleOpenMPStencil OpenMP_1D_stencil/1d_stencil)
        if (ACTIVEHARMONY_FOUND)
          set_tests_properties(ExampleOpenMPStencil PROPERTIES PASS_REGULAR_EXPRESSION "Test passed.")
          set_property(TEST ExampleOpenMPStencil PROPERTY ENVIRONMENT "APEX_POLICY=1")
          set_property(TEST ExampleOpenMPStencil APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_ENERGY=1")
          set_property(TEST ExampleOpenMPStencil APPEND PROPERTY ENVIRONMENT "APEX_THROTTLE_CONCURRENCY=1")
          set_property(TEST ExampleOpenMPStencil APPEND PROPERTY ENVIRONMENT "HARMONY_HOME=${ACTIVEHARMONY_ROOT}")
        endif()
        set_property (TEST ExampleOpenMPStencil
            APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PROCESSOR_COUNT}")
	  endif()
      if (NOT BUILD_STATIC_EXECUTABLES AND APEX_WITH_PLUGINS)
        add_test (ExampleOpenMP_Policy OpenMP_Policy/policy_test)
        set_property (TEST ExampleOpenMP_Policy
           APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PROCESSOR_COUNT}")
        set_property(TEST ExampleOpenMP_Policy APPEND PROPERTY ENVIRONMENT "APEX_OPENMP_VERBOSE=1")
        set_property(TEST ExampleOpenMP_Policy APPEND PROPERTY ENVIRONMENT "APEX_POLICY=1")
        set_property(TEST ExampleOpenMP_Policy APPEND PROPERTY ENVIRONMENT "HARMONY_HOME=${ACTIVEHARMONY_ROOT}")
        set_property(TEST ExampleOpenMP_Policy APPEND PROPERTY ENVIRONMENT "APEX_PLUGINS_PATH=${PROJECT_BINARY_DIR}/src/examples/OpenMP_Policy")
        set_property(TEST ExampleOpenMP_Policy APPEND PROPERTY ENVIRONMENT "APEX_PLUGINS=libapex_openmp_policy")
        set_property(TEST ExampleOpenMP_Policy APPEND PROPERTY ENVIRONMENT "APEX_OPENMP_SPACE=${PROJECT_SOURCE_DIR}/src/examples/OpenMP_Policy/space.json")
        set_property(TEST ExampleOpenMP_Policy APPEND PROPERTY ENVIRONMENT "LD_PRELOAD=${PROJECT_BINARY_DIR}/src/apex/libapex.so:${OMPT_LIBRARY}")
      endif()
    endif(PROCESSOR_COUNT)
  endif()
endif()

if (PAPI_FOUND AND NOT TAU_FOUND)
    set_property (TEST
        ExampleDemoC
        ExampleDemoCpp
        ExampleMatmult
        ExampleTestITT
        ExampleTestThreads
        ExampleCountCalls
        ExampleOverhead
        ExamplePolicyUnitTest
        ExamplePolicyEngineExample
        ExamplePolicyEngineCppExample
        ExampleReset
        ExampleGetPower
        ExampleThrottling
        ExampleThrottlingCpp
        APPEND PROPERTY ENVIRONMENT
         "APEX_PAPI_METRICS=PAPI_TOT_CYC PAPI_TOT_INS PAPI_LD_INS")
endif (PAPI_FOUND AND NOT TAU_FOUND)

