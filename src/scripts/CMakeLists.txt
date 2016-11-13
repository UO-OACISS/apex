CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/src/scripts/apex_exec
    ${CMAKE_BINARY_DIR}/src/scripts/apex_exec @ONLY)
#INSTALL_FILES(bin FILES ${CMAKE_BINARY_DIR}/src/scripts/apex_exec)


if (BUILD_STATIC_EXECUTABLES)
    INSTALL(FILES consolidate.py task_scatterplot.py DESTINATION bin
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
            GROUP_EXECUTE GROUP_READ
            WORLD_EXECUTE WORLD_READ)
else()
    INSTALL(FILES ${CMAKE_BINARY_DIR}/src/scripts/apex_exec apex_pthread_exec consolidate.py task_scatterplot.py DESTINATION bin
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
            GROUP_EXECUTE GROUP_READ
            WORLD_EXECUTE WORLD_READ)
endif()

