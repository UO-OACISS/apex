# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(APEX_GETINCLUDEDIRECTORY_LOADED TRUE)

macro(apex_get_include_directory variable)
  set(dir "")
  if(apex_SOURCE_DIR)
    set(dir "-I${apex_SOURCE_DIR}")
  elseif(APEX_ROOT)
    set(dir "-I${APEX_ROOT}/include")
  elseif($ENV{APEX_ROOT})
    set(dir "-I$ENV{APEX_ROOT}/include")
  endif()

  set(${variable} "${dir}")
endmacro()

###############################################################################
# prevent undefined variables from messing up the compilation flags
macro(apex_get_boost_include_directory variable)
  if(NOT BOOST_INCLUDE_DIR)
    set(${variable} "")
  else()
    set(${variable} "-I${BOOST_INCLUDE_DIR}")
  endif()
endmacro()

