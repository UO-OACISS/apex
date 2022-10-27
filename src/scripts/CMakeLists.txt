CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/src/scripts/apex_exec
	${PROJECT_BINARY_DIR}/src/scripts/apex_exec @ONLY)
#INSTALL_FILES(bin FILES ${PROJECT_BINARY_DIR}/src/scripts/apex_exec)


if (BUILD_STATIC_EXECUTABLES)
    INSTALL(FILES consolidate.py task_scatterplot.py counter_scatterplot.py gtrace_merger.py roofline_stats.py DESTINATION bin
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
            GROUP_EXECUTE GROUP_READ
            WORLD_EXECUTE WORLD_READ)
else()
	INSTALL(FILES ${PROJECT_BINARY_DIR}/src/scripts/apex_exec apex_pthread_exec consolidate.py task_scatterplot.py counter_scatterplot.py gtrace_merger.py roofline_stats.py DESTINATION bin
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
            GROUP_EXECUTE GROUP_READ
            WORLD_EXECUTE WORLD_READ)
endif()

