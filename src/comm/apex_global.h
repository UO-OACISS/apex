#ifndef APEX_GLOBAL_H
#define APEX_GLOBAL_H

/* required for Doxygen */
/** @file */

#include "apex.h"

#ifdef __cplusplus
extern "C"{
#endif 

/**
 \brief the function declaration, this is the funcion that does the reduction

 \param unused Unused value.
 \returns 0 on no error.
 */
int action_apex_reduce(void *unused);

/**
 \brief Each node has to populate their local value

 \param args Local data to be exchanged globally.
 \returns 0 on no error.
 */
int action_apex_get_value(void *args);

/**
 \brief A policy function to do periodic output

 \param context The context for the periodic policy.
 \returns 0 on no error.
 */
int apex_periodic_policy_func(apex_context const context) ;

/**
 \brief The function to set up global reductions

 \param in_action The address of a function.
 This is the function timer that should be reduced globally.
 This value is used for the example.
 \returns 0 on no error.
 */
void apex_global_setup(apex_function_address in_action);

/**
 \brief The function to tear down global reductions, if necessary
 */
void apex_global_teardown(void);

#ifdef __cplusplus
}
#endif 

#endif
