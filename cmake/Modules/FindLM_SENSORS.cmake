# - Try to find LM_SENSORS
# Once done this will define
#  LM_SENSORS_FOUND       - True if LM_SENSORS found.
#  LM_SENSORS_INCLUDE_DIR - where to find LM_SENSORS.h, etc.
#  LM_SENSORS_LIBRARIES   - List of libraries when using LM_SENSORS.


if(NOT DEFINED $LM_SENSORS_ROOT)
    if(DEFINED ENV{LM_SENSORS_ROOT})
    	# message("   env LM_SENSORS_ROOT is defined as $ENV{LM_SENSORS_ROOT}")
        set(LM_SENSORS_ROOT $ENV{LM_SENSORS_ROOT})
    endif()
endif()

find_path(LM_SENSORS_INCLUDE_DIR sensors/sensors.h
    HINTS ${LM_SENSORS_ROOT}/include
    /opt/local/include /usr/local/include /usr/include)

find_library(LM_SENSORS_LIBRARY NAMES sensors
    HINTS ${LM_SENSORS_ROOT}/lib
    /usr/lib /usr/lib64 /usr/local/lib /opt/local/lib)




include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LM_SENSORS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LM_SENSORS  DEFAULT_MSG
                                  LM_SENSORS_LIBRARY LM_SENSORS_INCLUDE_DIR)

mark_as_advanced(LM_SENSORS_INCLUDE_DIR LM_SENSORS_LIBRARY)

if (LM_SENSORS_FOUND)
  set(LM_SENSORS_LIBRARIES ${LM_SENSORS_LIBRARY} )
  set(LM_SENSORS_INCLUDE_DIRS ${LM_SENSORS_INCLUDE_DIR})
  set(LM_SENSORS_DIR ${LM_SENSORS_ROOT})
  add_definitions(-DAPEX_HAVE_LM_SENSORS)
  message(STATUS "Found LM_SENSORS: ${LM_SENSORS_LIBRARY}")
endif ()

