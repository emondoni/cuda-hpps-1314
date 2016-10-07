/*
 * simulate.h
 *
 *  Created on: 09/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef SIMULATE_H_
#define SIMULATE_H_

#include <curand_kernel.h>

/*** CONSTANT MEMORY DECLARATIONS ***/
extern __constant__ double d_sigma_wn; //noise parameter

/*** FUNCTION PROTOTYPES ***/
void perform_simulation(double * d_alpha, double * d_theta, double * d_time, double * d_matrix, double * d_periods, unsigned int noisy_flag, curandState * d_states);
__global__ void simulate(double * d_alpha, double * d_theta, double * d_time, double * d_matrix, double * d_periods, double * d_temp_vj);
__global__ void simulate_noisy(double * d_alpha, double * d_theta, double * d_time, double * d_matrix, double * d_periods, double * d_temp_vj, curandState * d_states);
__device__ double gamma_lut(double instant, double wave_period);
__device__ double vo_lut(double instant, double wave_period);

#endif /* SIMULATE_H_ */
