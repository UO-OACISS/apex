To run this example, you can either:

```bash
ctest -V -R ExamplePeriodicPlugin
```

or

```bash
export APEX_POLICY=1
export APEX_SCREEN_OUTPUT=1
export BUILDROOT=/path/to/build/directory
export APEX_PLUGINS_PATH=${BUILDROOT}/src/examples/PeriodicPlugin
export APEX_PLUGINS=libapex_periodic_policy
export DYLD_LIBRARY_PATH=${BUILDROOT}/src/apex
export DYLD_INSERT_LIBRARIES=${BUILDROOT}/src/apex/libapex.dylib:${BUILDROOT}/src/wrappers/libapex_pthread_wrapper.dylib
export DYLD_FORCE_FLAT_NAMESPACE=1
```

and you should see output like this:

```
khuck@Kevins-MacBook-Air build % ctest -V -R ExamplePeriodicPlugin
UpdateCTestConfiguration  from :/Users/khuck/src/xpress-apex/build/DartConfiguration.tcl
UpdateCTestConfiguration  from :/Users/khuck/src/xpress-apex/build/DartConfiguration.tcl
Test project /Users/khuck/src/xpress-apex/build
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end
test 82
    Start 82: ExamplePeriodicPlugin

82: Test command: /Users/khuck/src/xpress-apex/build/src/examples/PeriodicPlugin/periodic_policy_test
82: Environment variables:
82:  APEX_POLICY=1
82:  APEX_SCREEN_OUTPUT=1
82:  APEX_PLUGINS_PATH=/Users/khuck/src/xpress-apex/build/src/examples/PeriodicPlugin
82:  APEX_PLUGINS=libapex_periodic_policy
82:  DYLD_INSERT_LIBRARIES=/Users/khuck/src/xpress-apex/build/src/apex/libapex.dylib:/Users/khuck/src/xpress-apex/build/src/wrappers/libapex_pthread_wrapper.dylib
82:  DYLD_LIBRARY_PATH=/Users/khuck/src/xpress-apex/build/src/apex
82:  DYLD_FORCE_FLAT_NAMESPACE=1
82: Test timeout computed to be: 10000000
82: apex_plugin_init
82: apex_openmp_policy init
82: No iterations specified. Using default of 100.
82: IN MAIN
82: Iteration: 0
82: Iteration: 1
82: Iteration: 2
...
82: Iteration: 98
82: periodic_policy
82: Found 2 profiles so far.
82: pthread_join
82: void* std::__1::__thread_proxy<std::__1::tuple<std::__1::unique_ptr<std::__1::__thread_struct, std::__1::default_delete<std::__1::__thread_struct> >, void (*)(int), int> >(void*)
82: pthread_join : Num Calls : 392
82: pthread_join : Accumulated : 29.0294
82: pthread_join : Max : 0.539959
82:
82: Iteration: 99
82: periodic_policy
82: Found 2 profiles so far.
82: pthread_join
82: void* std::__1::__thread_proxy<std::__1::tuple<std::__1::unique_ptr<std::__1::__thread_struct, std::__1::default_delete<std::__1::__thread_struct> >, void (*)(int), int> >(void*)
82: pthread_join : Num Calls : 396
82: pthread_join : Accumulated : 29.5747
82: pthread_join : Max : 0.545258
82:
82: Test passed.
82:
82: apex_plugin_finalize
82: apex_openmp_policy finalize
82:
82: Elapsed time: 30.0875 seconds
82: Cores detected: 8
82: Worker Threads observed: 403
82: Available CPU time: 240.7 seconds
82:
82: Timer                                                : #calls  |    mean  |   total  |  % total
82: ------------------------------------------------------------------------------------------------
82: void* std::__1::__thread_proxy<std::__1::tuple<st... :      400      0.300    120.101     49.897
82: void* std::__1::__thread_proxy<std::__1::tuple<st... :        1     30.089     30.089     12.500
82:                                            APEX MAIN :        1     30.087     30.087    100.000
82:                                         pthread_join :      401      0.075     30.062     12.489
82:                                            APEX Idle :                         60.448     25.114
82: ------------------------------------------------------------------------------------------------
82:                                         Total timers : 802
1/1 Test #82: ExamplePeriodicPlugin ............   Passed   30.13 sec

The following tests passed:
	ExamplePeriodicPlugin
```
