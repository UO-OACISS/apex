#  Copyright (c) 2019 University of Oregon
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_library(hpxmp INTERFACE IMPORTED)
set_property(TARGET hpxmp PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${HPXMP_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/hpxmp/src)
set_property(TARGET hpxmp PROPERTY
    INTERFACE_LINK_LIBRARIES ${HPXMP_LIBRARIES})
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${HPXMP_LIBRARY_DIR})
message(INFO " Using binutils: ${HPXMP_LIBRARY_DIR} ${HPXMP_LIBRARIES}")

list(APPEND _apex_imported_targets hpxmp)

