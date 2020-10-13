#  Copyright (c) 2014 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(APEX_WITH_PLUGINS)
  message(INFO " apex will be built with plugin support.")
  set(LIBS ${LIBS} ${CMAKE_DL_LIBS})
  target_compile_definitions(apex_flags INTERFACE APEX_USE_PLUGINS)
endif()

# Regardless, APEX now depends on JSON for event filtering support.
# So make sure that we have the rapidjson library.

include(GitExternal)
  git_external(rapidjson
  https://github.com/miloyip/rapidjson.git
  master
  VERBOSE)

find_path(
  RAPIDJSON_INCLUDE_DIR
  NAMES rapidjson
  PATHS ${APEX_SOURCE_DIR}/rapidjson/include)

add_library(rapidjson INTERFACE IMPORTED)
if(RAPIDJSON_INCLUDE_DIR)
  message(INFO " Found rapidjson at ${RAPIDJSON_INCLUDE_DIR}")
  set_property(TARGET rapidjson PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${RAPIDJSON_INCLUDE_DIR})
  list(APPEND _apex_imported_targets rapidjson)
else()
  message(FATAL_ERROR " rapidjson not found. This should have been checked \
    out automatically. " "Try manually check out \
    https://github.com/miloyip/rapidjson.git to ${PROJECT_SOURCE_DIR}")
endif()
