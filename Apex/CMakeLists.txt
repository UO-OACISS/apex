# Create a library called "Apex" which includes the source file "apex.cxx". 
# The extension is already found. Any number of sources could be listed here. 
add_library (ApexStatic STATIC Apex.cpp ThreadInstance.cpp EventListener.cpp Handler.cpp ConcurrencyHandler.cpp PolicyHandler.cpp TauListener.cpp)
add_library (ApexShared SHARED Apex.cpp ThreadInstance.cpp EventListener.cpp Handler.cpp ConcurrencyHandler.cpp PolicyHandler.cpp TauListener.cpp)
set_target_properties(ApexStatic PROPERTIES OUTPUT_NAME Apex)
set_target_properties(ApexShared PROPERTIES OUTPUT_NAME Apex)

target_link_libraries(ApexStatic ${LIBS})
target_link_libraries(ApexShared ${LIBS})

# add the binary tree to the search path for include files
# so that we will find ApexConfig.h
include_directories("${PROJECT_BINARY_DIR}/Apex")

INSTALL(FILES apex.h apex.hpp Apex.hpp Handler.hpp EventListener.hpp DESTINATION include)

INSTALL(TARGETS ApexShared ApexStatic
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

