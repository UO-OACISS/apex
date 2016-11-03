//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef APEX_HAVE_HPX
#include <hpx/config.hpp>

#include "apex.hpp"
#include "handler.hpp"

namespace apex {

boost::asio::io_service handler::_io;

}

#endif
