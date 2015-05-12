# - Find LM_SENSORS
# Find the native LM_SENSORS includes and library
#
#  LM_SENSORS_INCLUDE_DIR - where to find LM_SENSORS.h, etc.
#  LM_SENSORS_LIBRARIES   - List of libraries when using LM_SENSORS.
#  LM_SENSORS_FOUND       - True if LM_SENSORS found.


if (LM_SENSORS_INCLUDE_DIR)
  # Already in cache, be silent
  set(LM_SENSORS_FIND_QUIETLY TRUE)
endif ()

if(NOT DEFINED $LM_SENSORS_ROOT)
    if(DEFINED ENV{LM_SENSORS_ROOT})
        set(LM_SENSORS_ROOT $ENV{LM_SENSORS_ROOT})
    endif()
endif()

find_path(LM_SENSORS_INCLUDE_DIR sensors/sensors.h
    ${LM_SENSORS_ROOT}/include
  /opt/local/include
  /usr/local/include
  /usr/include
)

set(TMP_PATH $ENV{LD_LIBRARY_PATH})
if ($TM_PATH)
  string(REPLACE ":" " " LD_LIBRARY_PATH_STR $TMP_PATH)
endif()
set(LM_SENSORS_NAME sensors)
find_library(LM_SENSORS_LIBRARY
  NAME ${LM_SENSORS_NAME}
  PATHS ${LM_SENSORS_ROOT}/lib /usr/lib /usr/lib64 /usr/local/lib /opt/local/lib ${LD_LIBRARY_PATH_STR}
)

if (LM_SENSORS_INCLUDE_DIR AND LM_SENSORS_LIBRARY)
   set(LM_SENSORS_FOUND TRUE)
    set( LM_SENSORS_LIBRARIES ${LM_SENSORS_LIBRARY} )
else ()
   set(LM_SENSORS_FOUND FALSE)
   set( LM_SENSORS_LIBRARIES )
endif ()

if (LM_SENSORS_FOUND)
   if (NOT LM_SENSORS_FIND_QUIETLY)
      message(STATUS "Found LM_SENSORS: ${LM_SENSORS_LIBRARY}")
   endif ()
   add_definitions(-DAPEX_HAVE_LM_SENSORS)
else ()
      message(STATUS "Not Found LM_SENSORS: ${LM_SENSORS_LIBRARY}")
   if (LM_SENSORS_FIND_REQUIRED)
      message(STATUS "Looked for LM_SENSORS libraries named ${LM_SENSORSS_NAME}.")
      message(FATAL_ERROR "Could NOT find LM_SENSORS library")
   endif ()
endif ()

mark_as_advanced(
  LM_SENSORS_LIBRARY
  LM_SENSORS_INCLUDE_DIR
)
