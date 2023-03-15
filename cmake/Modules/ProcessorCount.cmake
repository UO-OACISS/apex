if(NOT DEFINED PROCESSOR_COUNT)
  # Unknown:
  set(PROCESSOR_COUNT 1)

  # Linux:
  set(cpuinfo_file "/proc/cpuinfo")
  if(EXISTS "${cpuinfo_file}")
    file(STRINGS "${cpuinfo_file}" procs REGEX "^processor.: [0-9]+$")
    list(LENGTH procs PROCESSOR_COUNT)
  endif()

  # Mac:
  if(APPLE)
    find_program(cmd_sys_pro "sysctl")
    if(cmd_sys_pro)
      execute_process(COMMAND ${cmd_sys_pro} -n hw.ncpu OUTPUT_VARIABLE
      PROCESSOR_COUNT OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
  endif()

  # Windows:
  if(WIN32)
    set(PROCESSOR_COUNT "$ENV{NUMBER_OF_PROCESSORS}")
  endif()
endif()

if (DEFINED PROCESSOR_COUNT)
    message(INFO " Found ${PROCESSOR_COUNT} cores.")
    if (PROCESSOR_COUNT GREATER 16)
        set(PROCESSOR_COUNT 16)
    endif (PROCESSOR_COUNT GREATER 16)
    message(INFO " Using ${PROCESSOR_COUNT} cores.")
endif(DEFINED PROCESSOR_COUNT)
