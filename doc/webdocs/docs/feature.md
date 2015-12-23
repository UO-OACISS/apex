# Motivation

Frequently, software components or even entire applications run into a situation where the context of the execution environment has changed in some way (or does not meet assumptions). In those situations, the software requires some mechanism for evaluating its own performance and that of the underlying runtime system, operating system and hardware. The types of adaptation that the software wants to do could include:

* Controlling concurrency
	* to improve energy efficiency
	* for performance
* Parametric variability
	* adjust the decomposition granularity for this machine / dataset
	* choose a different algorithm for better performance/accuracy
	* choose a different preconditioner for better performance/accuracy
	* choose a different solver for better performance/accuracy
* Load Balancing
	* when to perform AGAS migration?
	* when to perform repartitioning?
	* when to perform data exchanges?
* Parallel Algorithms (for_each…) - choose a different execution model
	* separate *what* from *how*
* Address the “SLOW(ER)” performance model
	* avoid **S**tarvation
	* reduce **L**atency
	* reduce **O**verhead
	* reduce **W**aiting
	* reduce **E**nergy consumption
	* improve **R**esiliency

APEX provides both *performance awareness* and *performance adaptation*.

* APEX provides top-down and bottom-up performance mapping and feedback.
* APEX exposes node-wide resource utilization data and analysis, energy consumption, and health information in real time
	* Software can subsequently associate performance state with policy for feedback control
* APEX introspection
	* OS: track system resources, utilization, job contention, overhead
	* Runtime (HPX, HPX-5, OpenMP...): track threads, queues, concurrency, remote operations, parcels, memory management
	* Application timer / counter observation

![Screenshot](img/APEX_diagram.pdf)

*Above: APEX architecture diagram. The application sends events to the APEX instrumentation API, which updates the performance state. The Policy Engine executes policies that change application behavior based on rule outcomes.*

## Introspection

APEX collects data through *inspectors*. The synchronous data collection uses an event API and event *listeners*. The API includes events for:

* Initialize, terminate, new thread 
	* added to the HPX thread scheduler
	* added to the HPX-5 thread scheduler
	* added to the OpenMP runtime using the OMPT interface
	* added to the pthread runtime by wrapping the pthread API calls
* Timer start, stop, yield, resume 
	* added to HPX task scheduler
	* added to HPX-5 task scheduler
	* added to the OpenMP runtime using the OMPT interface
	* added to the pthread runtime by wrapping the pthread API calls
* Sampled values
	* counters from HPX, HPX-5
* Custom events (meta-events)
  * useful for triggering policies

Asynchonous data collection does not rely on events, but occurs periodically.  APEX exploits access to performance data from lower stack components (i.e. the runtime) or by reading from the RCR blackboard (i.e., power, energy). Other operating system and hardware health data is collected through other interfaces:

* /proc/stat
* /proc/cpuinfo
* /proc/meminfo
* /proc/net/dev
* /proc/self/status
* lm_sensors
* power measurements 

## Event Listeners

There are a number of listeners in APEX that are triggered by the events passed in through the API. For example, the **Profiling Listener** records events related to maintaining the performance state.

* Start Event: records the name/address of the timer, gets a timestamp (using rdtsc), returns a profiler handle
* Stop Event: gets a timestamp, puts the profiler object in a queue for back-end processing and returns
* Sample Event: put the name & value in the queue

Internally to APEX, there is an asynchronous consumer thread that processes profiler objects and samples to build a performance profile (in HPX, this thread is processed/scheduled as an HPX thread/task).

The TAU Listener (used for postmortem analysis) synchronously passes all measurement events to TAU to build an offline profile or trace. TAU will also capture any other events for which it is configured, including MPI, memory, file I/O, etc.

The concurrency listener (also used for postmortem analysis) maintains a timeline of total concurrency, periodically sampled from  within APEX.

* Start event: push timer ID on stack
* Stop event: pop timer ID off stack

An asynchronous consumer thread periodically logs the current timer for each thread. This thread will output a concurrency data report and gnuplot script at APEX termination. 

## Policy Listener

Policies are rules that decide on outcomes based on observed state. 
Triggered policies are invoked by introspection API events.
Periodic policies are run periodically on asynchronous thread.
Polices are registered with the Policy Engine at program startup by runtime code and/or from the application.
Applications, runtimes, and the OS can register callback functions to be executed. 
Callback functions define the policy rules - “If x < y then...(take some action!)”.

* Enables runtime adaptation using introspection data
* Engages actuators across stack layers
* Is also used to involve online auto-tuning support




