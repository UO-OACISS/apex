# Copyright (c) 2014-2018 Kevin Huck
# Copyright (c) 2014-2018 University of Oregon
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


if(HPX_WITH_APEX)
    include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.hpx)
    return()
else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.standalone)
    return()
endif()
