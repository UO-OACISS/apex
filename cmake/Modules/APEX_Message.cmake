# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(APEX_MESSAGE_LOADED TRUE)

macro(apex_info type)
  string(TOLOWER ${type} lctype)
  message("[apex.info.${lctype}] " ${ARGN})
endmacro()

macro(apex_debug type)
  if("${APEX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    string(TOLOWER ${type} lctype)
    message("[apex.debug.${lctype}] " ${ARGN})
  endif()
endmacro()

macro(apex_warn type)
  if("${APEX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug|WARN|warn|Warn")
    string(TOLOWER ${type} lctype)
    message("[apex.warn.${lctype}] " ${ARGN})
  endif()
endmacro()

macro(apex_error type)
  string(TOLOWER ${type} lctype)
  message(FATAL_ERROR "[apex.error.${lctype}] " ${ARGN})
endmacro()

macro(apex_message level type)
  if("${level}" MATCHES "ERROR|error|Error")
    string(TOLOWER ${type} lctype)
    apex_error(${lctype} ${ARGN})
  elseif("${level}" MATCHES "WARN|warn|Warn")
    string(TOLOWER ${type} lctype)
    apex_warn(${lctype} ${ARGN})
  elseif("${level}" MATCHES "DEBUG|debug|Debug")
    string(TOLOWER ${type} lctype)
    apex_debug(${lctype} ${ARGN})
  elseif("${level}" MATCHES "INFO|info|Info")
    string(TOLOWER ${type} lctype)
    apex_info(${lctype} ${ARGN})
  else()
    apex_error("message" "\"${level}\" is not an APEX configuration logging level.")
  endif()
endmacro()

macro(apex_config_loglevel level return)
  set(${return} FALSE)
  if(    "${APEX_CMAKE_LOGLEVEL}" MATCHES "ERROR|error|Error"
     AND "${level}" MATCHES "ERROR|error|Error")
    set(${return} TRUE)
  elseif("${APEX_CMAKE_LOGLEVEL}" MATCHES "WARN|warn|Warn"
     AND "${level}" MATCHES "WARN|warn|Warn")
    set(${return} TRUE)
  elseif("${APEX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug"
     AND "${level}" MATCHES "DEBUG|debug|Debug")
    set(${return} TRUE)
  elseif("${APEX_CMAKE_LOGLEVEL}" MATCHES "INFO|info|Info"
     AND "${level}" MATCHES "INFO|info|Info")
    set(${return} TRUE)
  endif()
endmacro()

macro(apex_print_list level type message list)
  apex_config_loglevel(${level} printed)
  if(printed)
    if(${list})
      apex_message(${level} ${type} "${message}: ")
      foreach(element ${${list}})
        message("    ${element}")
      endforeach()
    else()
      apex_message(${level} ${type} "${message} is empty.")
    endif()
  endif()
endmacro()

