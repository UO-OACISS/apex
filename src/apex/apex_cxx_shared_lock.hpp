//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if __cplusplus > 201701L 
#include <shared_mutex>
#elif __cplusplus > 201402L
#include <shared_mutex>
#else
#include <mutex>
#endif

namespace apex
{

#if __cplusplus > 201701L 
    typedef std::shared_mutex shared_mutex_type;
    typedef std::shared_lock<shared_mutex_type> read_lock_type;
    typedef std::unique_lock<shared_mutex_type> write_lock_type;
#elif __cplusplus > 201402L
    typedef std::shared_timed_mutex shared_mutex_type;
    //typedef std::mutex shared_mutex_type;
    typedef std::unique_lock<shared_mutex_type> read_lock_type;
    typedef std::unique_lock<shared_mutex_type> write_lock_type;
#else
    typedef std::mutex shared_mutex_type;
    typedef std::unique_lock<shared_mutex_type> read_lock_type;
    typedef std::unique_lock<shared_mutex_type> write_lock_type;
#endif

}

