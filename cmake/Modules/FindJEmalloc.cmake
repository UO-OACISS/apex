# - Find JEmalloc
# Find the native JEmalloc includes and library
#
#  JEmalloc_INCLUDE_DIR - where to find JEmalloc.h, etc.
#  JEmalloc_LIBRARIES   - List of libraries when using JEmalloc.
#  JEmalloc_FOUND       - True if JEmalloc found.


if (JEmalloc_INCLUDE_DIR)
  # Already in cache, be silent
  set(JEmalloc_FIND_QUIETLY TRUE)
endif ()

if(NOT DEFINED $JEMALLOC_ROOT)
    if(DEFINED ENV{JEMALLOC_ROOT})
        set(JEMALLOC_ROOT $ENV{JEMALLOC_ROOT})
    endif()
endif()

find_path(JEmalloc_INCLUDE_DIR jemalloc/jemalloc.h
  ${JEMALLOC_ROOT}/include
  /opt/local/include
  /usr/local/include
  /usr/include
)

set(TMP_PATH $ENV{LD_LIBRARY_PATH})
if ($TM_PATH)
  string(REPLACE ":" " " LD_LIBRARY_PATH_STR $TMP_PATH)
endif()
set(JEmalloc_NAME jemalloc)
find_library(JEmalloc_LIBRARY
  NAME ${JEmalloc_NAME}
  PATHS ${JEMALLOC_ROOT}/lib /usr/lib /usr/local/lib /opt/local/lib ${LD_LIBRARY_PATH_STR}
)

if (JEmalloc_INCLUDE_DIR AND JEmalloc_LIBRARY)
   set(JEmalloc_FOUND TRUE)
    set( JEmalloc_LIBRARIES ${JEmalloc_LIBRARY} )
else ()
   set(JEmalloc_FOUND FALSE)
   set( JEmalloc_LIBRARIES )
endif ()

if (JEmalloc_FOUND)
   if (NOT JEmalloc_FIND_QUIETLY)
      message(STATUS "Found JEmalloc: ${JEmalloc_LIBRARY}")
   endif ()
else ()
      message(STATUS "Not Found JEmalloc: ${JEmalloc_LIBRARY}")
   if (JEmalloc_FIND_REQUIRED)
      message(STATUS "Looked for JEmalloc libraries named ${JEmallocS_NAME}.")
      message(FATAL_ERROR "Could NOT find JEmalloc library")
   endif ()
endif ()

mark_as_advanced(
  JEmalloc_LIBRARY
  JEmalloc_INCLUDE_DIR
)
