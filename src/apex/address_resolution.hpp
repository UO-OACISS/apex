//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef ADDRESS_RESOLUTION_HPP
#define ADDRESS_RESOLUTION_HPP

#ifdef APEX_HAVE_HPX3
#include <hpx/config.hpp>
#endif

namespace apex {
    std::string * lookup_address(uintptr_t ip, bool withFileInfo);
}

#endif // ADDRESS_RESOLUTION_HPP
