# APEX Specification *...DRAFT!...*


This page contains the API specification for APEX. The API specification provides a high-level overview of the API and its functionality. For full implementation details, please see the [API Reference Manual](refman.md).  The following API strings reflect the C++ and C names. The C++ names will use overloading for different argument lists, and will replace the "apex\_" prefix with the "apex::" namespace.

## General Utility functions

### apex::init (thread_name)<br>apex::init (argc, argv, thread\_name)<br>apex\_init (thread\_name)<br>apex\_init\_args (argc, argv, thread\_name)

APEX initialization is required to set up data structures and spawn the necessary helper threads, including the background system state query thread, the policy engine thread, and the profile handler thread. The thread name parameter will be used as the top-level timer for the the main thread of execution. The arguments argc and argv are optional, and would be passed from the main function of the appliation or runtime.

### apex::finalize ()<br>apex\_finalize ()

APEX finalization is required to format any desired output (screen, csv, profile, etc.) and terminate all APEX helper threads. No memory is freed at this point - that is done by the apex\_cleanup() call.  The reason for this is that applications may want to perform reporting after finalization, so the performance state of the application should still exist. 

### apex::cleanup ()<br>apex\_cleanup ()

APEX cleanup frees all memory associated with APEX. 

### apex::set\_node\_id (id)<br>apex\_set\_node\_id (id)

When running in distributed environments, assign the specified id number as the APEX node ID. This can be an MPI rank or an HPX locality, for example. 

### apex::register\_thread (name)<br>apex\_register\_thread (name)

Register a new OS thread with APEX. The thread name parameter will be used as the top-level timer for the the new thread of execution.  This method should be called whenever a new OS thread is spawned by the application or the runtime.

### apex::exit\_thread ()<br>apex\_exit\_thread ()

Before any thread other than the main thread of execution exits, notify APEX that the thread is exiting. The main thread should not call this function, but apex\_finalize instead.

### apex::version ()<br> apex\_version ()

Return the APEX version as a string.

### apex::print\_options ()<br> apex\_print\_options ()

Print the current APEX options to stdout.

## Basic Measurement Functions (timers, counters)

### apex::start (timer\_name)<br>apex::start (function\_address)<br>apex\_start (type, identifier)

Create an APEX timer and start it. An APEX profiler object is returned, containing an identifier that APEX uses to stop the timer.  The timer is either identified by a name or a function/task instruction pointer address.

### apex::stop (the\_profiler)<br>apex\_stop (the\_profiler)

The timer associated with the profiler object is stopped and placed on an internal queue to be processed by the profiler handler thread in the background.  The profiler object is flagged as "stopped", so that when the profiler is processed the call count for this particular timer will be incremented by 1.

### apex::yield (the\_profiler)<br>apex\_yield (the\_profiler)

The timer associated with the profiler object is stopped and placed on an internal queue to be processed by the profiler handler thread in the background.  The profiler object is flagged as "NOT stopped", so that when the profiler is processed the call count will not be incremented.  An application using apex\_yield should not use apex\_resume to restart the timer, it should use apex\_start.

### apex::resume (timer\_name)<br>apex::resume(function\_address)<br>apex\_resume (type, identifier)

Create an APEX timer and start it. An APEX profiler object is returned, containing an identifier that APEX uses to stop the timer.  The profiler is flagged as "NOT a new task", so that when it is stopped by apex\_stop the call count for this particular timer will not be incremented.

### apex::new\_task (name, task\_info)<br>apex::new\_task (function\_address, task\_info)<br>apex\_new\_tasks (type, identifier, task\_info)

Register the creation of a new task. This is used to track task dependencies in APEX. APEX assumes that the current APEX profiler refers to the task that is the parent of this new task. The task\_info object is a generic pointer to whatever data might need to be passed to a policy executed on when a new task is created.

### apex::sample\_value (name, value)<br>apex\_sample\_value (name, value)

Record a measurement of the specified counter with the specified value. For example, "bytes transferred" and "1024".

### apex::set\_state (state)<br>apex\_set\_state (state)

Set the state of the current OS thread.  States can include things like idle, running, blocked, etc.

## Policy related querying and modifying methods

### apex::register\_policy (event\_type, function)<br>apex::register\_policy (std::set&lt;event\_type&gt;, function)<br>apex\_register\_policy (event\_type, function)

Apex provides the ability to call an application-specified function when certain events occur in the APEX library, or periodically. This assigns the passed in function to the event, so that when that event occurs in APEX, the function is called. The context for the event will be passed to the registered function.

### apex::register\_periodic\_policy (period, function)<br>apex\_register\_periodic\_policy (period, function)

Apex provides the ability to call an application-specified function when certain events occur in the APEX library, or periodically. This method assigns the passed in function to be called on a periodic basis. The context for the event will be passed to the registered function.  The period units are in microseconds (us).

### apex::deregister\_policy (handle)<br>apex\_deregister\_policy (handle)

Remove the specified policy so that it will no longer be executed, whether it is event-based or periodic.

### apex::register\_custom\_event (name)<br>apex\_register\_custom\_event (name)

Register a new event type with APEX.

### apex::custom\_event (name, event\_data)<br>apex\_custom\_event (name, event\_data)

Trigger a custom event.  This function will pass a custom event to the APEX event listeners. Each listeners' custom event handler will handle the custom event. Policy functions will be passed the custom event name in the event context.

### apex::get\_profile (name)<br>apex::get\_profile (function\_address)<br>apex\_get\_profile (type, identifier)

This function will return the current profile for the specified identifier. Because profiles are updated out-of-band, it is possible that this profile value is out of date. This profile can be either a timer or a sampled value.

### apex::reset (timer_name)<br>apex::reset (function\_address)<br>apex\_reset (type, identifier)

This function will reset the profile associated with the specified timer or counter id to zero.  If the identifier is null, all timers and counters will be reset.

## Concurrency Throttling Policy Functions

### apex::setup\_power\_cap\_throttling ()<br>apex\_setup\_power\_cap\_throttling ()

This function will initialize APEX for power cap throttling. With power cap throttling, APEX will reduce the thread concurrency to keep the application under a specified power cap.  There are several environment variables that control power cap throttling:

* APEX\_THROTTLE\_CONCURRENCY
* APEX\_THROTTLE\_ENERGY
* APEX\_THROTTLING\_MAX\_THREADS
* APEX\_THROTTLING\_MIN\_THREADS
* APEX\_THROTTLING\_MAX\_WATTS
* APEX\_THROTTLING\_MIN\_WATTS

### apex::setup\_timer\_throttling (timer\_name, criteria, method, update\_interval)<br>apex::setup\_timer\_throttling (function\_address, criteria, method, update\_interval)<br>apex\_setup\_timer\_throttling (type, identifier, criteria, method, update\_interval)

This function will initialize the throttling policy to optimize for the specified function. The optimization criteria include maximizing throughput, minimizing or maximizing time spent in the specified function. After evaluating the state of the system, the policy will set the thread cap, which can be queried using apex\_get\_thread\_cap().

### apex::setup\_throughput\_throttling (timer\_name, criteria, event\_type, num\_inputs, inputs, mins, maxs, steps)<br>apex::setup\_throughput\_throttling (function\_address, criteria, event\_type, num\_inputs, inputs, mins, maxs, steps)<br>apex\_setup\_throughput\_throttling (type, identifier, criteria, event\_type, num\_inputs, inputs, mins, maxs, steps)

This function will initialize the throttling policy to optimize for the specified function. The optimization criteria include maximizing throughput, minimizing or maximizing time spent in the specified function. After evaluating the state of the system, the policy will set the thread cap, which can be queried using apex\_get\_thread\_cap().

### apex::setup\_custom\_tuning (std::function&lt; double(void)&gt;, event\_type, num\_inputs, inputs, mins, maxs, steps)

Setup tuning of specified parameters to optimize for a custom metric, using multiple input criteria.  This function will initialize a policy to optimize a custom metric, using the list of tunable parameters. The system tries to minimize the custom metric. After evaluating the state of the system, the policy will assign new values to the inputs.

### apex::get\_tunable\_params (handle)

Return a vector of the current tunable parameters.

### apex::shutdown\_throttling ()<br>apex\_shutdown\_throttling ()

This function will terminate the throttling policy.

### apex::get\_thread\_cap ()<br>apex\_get\_thread\_cap ()

This function will return the current thread cap based on the throttling policy.

### apex::set\_thread\_cap (new\_cap)<br>apex\_set\_thread\_cap (new\_cap)

This function will set the current thread cap based on an external throttling policy.

## Event-based API (OCR, Legion - _TBD_)

### apex::task\_create (parent\_id)
### apex::dependency\_reached (event\_id, data\_id, task\_id, parent\_id, ?)
### apex::task\_ready (why\_ready)
### apex::task\_execute (why\_delay, function)
### apex::task\_finished ()
### apex::task\_destroy ()

### apex::data\_create ()
### apex::data\_new\_size ()
### apex::data\_move\_from ()
### apex::data\_move\_to ()
### apex::data\_replace ()
### apex::data\_destroy ()

### apex::event\_create ()
### apex::event\_add\_dependency ()
### apex::event\_trigger ()
### apex::event\_destroy ()
