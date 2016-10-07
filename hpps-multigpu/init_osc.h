/*
 * init_osc.h
 *
 *  Created on: 04/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef INIT_OSC_H_
#define INIT_OSC_H_

#include "samples.h"
#include <curand_kernel.h>

/*** MACRO DEFINITIONS ***/
#define N_INTERVALS N_SAMPLES //the number of time intervals equals that of the samples in a period
#define RAND_COEF 1e-04 //the order of magnitude of the discrepancy introduced by the randomization of the periods

/*** FUNCTION PROTOTYPES ***/
void initialize_simulation(double ** d_periods, double ** d_time, curandState *d_states);
__global__ void randomize_periods(double * d_periods, curandState * d_states);
__global__ void generate_time_vector(double * d_time);

#endif /* INIT_OSC_H_ */
