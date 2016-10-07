/*
 * init_output.h
 *
 *  Created on: 06/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef INIT_OUTPUT_H_
#define INIT_OUTPUT_H_

#include <curand_kernel.h>

/*** FUNCTION PROTOTYPES ***/
void initialize_output(double ** d_alpha, double ** d_theta, double * d_periods, curandState * d_states);
__global__ void randomize_t0(double * d_alpha, double * d_theta, double * d_periods, curandState * d_states);

#endif /* INIT_OUTPUT_H_ */
