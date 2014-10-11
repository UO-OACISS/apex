# Create a library called "Apex" which includes the source file "apex.cxx". 
# The extension is already found. Any number of sources could be listed here. 

if (TAU_FOUND)
add_library (apexStatic STATIC apex.cpp thread_instance.cpp event_listener.cpp handler.cpp concurrency_handler.cpp policy_handler.cpp tau_listener.cpp RCR.bb.cpp)
add_library (apexShared SHARED apex.cpp thread_instance.cpp event_listener.cpp handler.cpp concurrency_handler.cpp policy_handler.cpp tau_listener.cpp RCR.bb.cpp)
else()
add_library (apexStatic STATIC apex.cpp thread_instance.cpp event_listener.cpp handler.cpp concurrency_handler.cpp policy_handler.cpp RCR.bb.cpp profiler_listener.cpp)
add_library (apexShared SHARED apex.cpp thread_instance.cpp event_listener.cpp handler.cpp concurrency_handler.cpp policy_handler.cpp RCR.bb.cpp profiler_listener.cpp)
endif()

set_target_properties(apexStatic PROPERTIES OUTPUT_NAME apex)
set_target_properties(apexShared PROPERTIES OUTPUT_NAME apex)

target_link_libraries(apexStatic ${LIBS} ${Boost_LIBRARIES})
target_link_libraries(apexShared ${LIBS} ${Boost_LIBRARIES})


# add the binary tree to the search path for include files
# so that we will find ApexConfig.h
include_directories("${PROJECT_BINARY_DIR}/apex")

INSTALL(FILES apex.h apex.hpp handler.hpp event_listener.hpp DESTINATION include)

INSTALL(TARGETS apexShared apexStatic
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

