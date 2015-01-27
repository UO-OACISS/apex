#ifndef APEX_GLOBAL_H
#define APEX_GLOBAL_H

#include "apex.h"
#include "math.h"
#include "stdio.h"

// every node has the local value
extern apex_profile value;

// root node has the reduced value
extern apex_profile reduced_value;

// declare a pointer to the callback function
extern apex_function_address action_apex_allreduce_handle;

#ifdef __cplusplus
extern "C"{
#endif 

// the function declaration, this is the funcion that does the reduction
int action_apex_reduce(void *unused);

// Each node has to populate their local value
int action_apex_get_value(void *args);

// each node has to ...?
int action_apex_set_value(void *args);

// The function to do periodic output
int apex_periodic_output(apex_context const context) ;

// The function to set up global reductions
void apex_global_setup(apex_function_address in_action);

#ifdef __cplusplus
}
#endif 

#endif
