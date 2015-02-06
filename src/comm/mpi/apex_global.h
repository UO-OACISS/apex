#ifndef APEX_GLOBAL_H
#define APEX_GLOBAL_H

#include "apex.h"

#ifdef __cplusplus
extern "C"{
#endif 

// the function declaration, this is the funcion that does the reduction
int action_apex_reduce(void *unused);

// Each node has to populate their local value
int action_apex_get_value(void *args);

// The function to do periodic output
int apex_periodic_output(apex_context const context) ;

// The function to set up global reductions
void apex_global_setup(apex_function_address in_action);

// The function to tear down global reductions
void apex_global_teardown(void);

#ifdef __cplusplus
}
#endif 

#endif
