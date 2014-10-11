//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// apex main class
#ifndef APEX_TYPES_HPP
#define APEX_TYPES_HPP

/* Typedef for enumerating the different event types */

typedef enum _event_type {
  STARTUP,
  SHUTDOWN,
  NEW_NODE,
  NEW_THREAD,
  START_EVENT,
  RESUME_EVENT,
  STOP_EVENT,
  SAMPLE_VALUE,
  PERIODIC
} apex_event_type;

typedef struct _policy_handle
{
    int id;
} apex_policy_handle;

typedef struct _context
{
    apex_event_type event_type;
    apex_policy_handle* policy_handle;
} apex_context;

#endif //APEX_TYPES_HPP
