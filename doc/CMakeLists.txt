# add a target to generate API documentation with Doxygen
find_package(Doxygen)
find_package(LATEX COMPONENTS PDFLATEX MAKEINDEX)

if(DOXYGEN_FOUND)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(doc ${DOXYGEN_EXECUTABLE}
		${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY
		${CMAKE_CURRENT_BINARY_DIR} 
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
	INSTALL( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION doc )
    if(DEFINED PDFLATEX_COMPILER)
        MESSAGE(STATUS "Enabled: building PDF reference manual")
        add_custom_command(
            OUTPUT   ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf 
            COMMAND  ${CMAKE_MAKE_PROGRAM}
            DEPENDS  #${CMAKE_CURRENT_BINARY_DIR}/latex/refman.tex
            ${CMAKE_CURRENT_BINARY_DIR}/latex/Makefile
            COMMENT  "Creating PDF of documentation"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/latex
        )
        add_custom_target(refman.pdf
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf
            COMMENT  "Executing ${PDFLATEX_COMPILER}"
        )
	    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf DESTINATION doc )
    endif()
else()
    MESSAGE(STATUS "Doxygen not found. Building documentation requires Doxygen.")
endif(DOXYGEN_FOUND)

