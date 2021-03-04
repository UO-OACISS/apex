To run this example, you can either:

```bash
ctest -V -R ExamplePeriodicPlugin
```

or

```bash
export APEX_POLICY=1
export BUILDROOT=/path/to/build/directory
export APEX_PLUGINS_PATH=${BUILDROOT}/src/examples/PeriodicPlugin
export APEX_PLUGINS=libapex_periodic_policy
export DYLD_LIBRARY_PATH=${BUILDROOT}/src/apex
export DYLD_INSERT_LIBRARIES=${BUILDROOT}/src/apex/libapex.so
export DYLD_FORCE_FLAT_NAMESPACE=1
```