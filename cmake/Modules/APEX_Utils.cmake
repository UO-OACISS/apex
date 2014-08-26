## force out of tree build

set(APEX_FORCEOUTOFTREEBUILD_LOADED TRUE)

include(APEX_Include)

apex_include(Message)

macro(apex_force_out_of_tree_build message)
  string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" insource)
  get_filename_component(parentdir ${CMAKE_SOURCE_DIR} PATH)
  string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${parentdir}" insourcesubdir)
  if(insource OR insourcesubdir)
    apex_error("in_tree" "${message}")
  endif()
endmacro()


