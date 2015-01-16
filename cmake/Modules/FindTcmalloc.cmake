# - Find Tcmalloc
# Find the native Tcmalloc includes and library
#
#  Tcmalloc_INCLUDE_DIR - where to find Tcmalloc.h, etc.
#  Tcmalloc_LIBRARIES   - List of libraries when using Tcmalloc.
#  Tcmalloc_FOUND       - True if Tcmalloc found.


if (Tcmalloc_INCLUDE_DIR)
  # Already in cache, be silent
  set(Tcmalloc_FIND_QUIETLY TRUE)
endif ()

find_path(Tcmalloc_INCLUDE_DIR google/heap-checker.h
  ${GPERFTOOLS_ROOT}/include
  /opt/local/include
  /usr/local/include
  /usr/include
)

string(REPLACE ":" " " LD_LIBRARY_PATH_STR $ENV{LD_LIBRARY_PATH})
set(Tcmalloc_NAME tcmalloc)
find_library(Tcmalloc_LIBRARY
  NAME ${Tcmalloc_NAME}
  PATHS ${GPERFTOOLS_ROOT}/lib /usr/lib /usr/local/lib /opt/local/lib ${LD_LIBRARY_PATH_STR}
)

if (Tcmalloc_INCLUDE_DIR AND Tcmalloc_LIBRARY)
   set(Tcmalloc_FOUND TRUE)
    set( Tcmalloc_LIBRARIES ${Tcmalloc_LIBRARY} )
else ()
   set(Tcmalloc_FOUND FALSE)
   set( Tcmalloc_LIBRARIES )
endif ()

if (Tcmalloc_FOUND)
   if (NOT Tcmalloc_FIND_QUIETLY)
      message(STATUS "Found Tcmalloc: ${Tcmalloc_LIBRARY}")
   endif ()
else ()
      message(STATUS "Not Found Tcmalloc: ${Tcmalloc_LIBRARY}")
   if (Tcmalloc_FIND_REQUIRED)
      message(STATUS "Looked for Tcmalloc libraries named ${TcmallocS_NAME}.")
      message(FATAL_ERROR "Could NOT find Tcmalloc library")
   endif ()
endif ()

mark_as_advanced(
  Tcmalloc_LIBRARY
  Tcmalloc_INCLUDE_DIR
)
