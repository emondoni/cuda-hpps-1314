/*
 * init_output.cu
 *
 *  Created on: 06/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "init_output.h"
#include "sim_parameters.h"
#include "cuda_settings.h"
#include "rng.h"
#include <curand_kernel.h>

void initialize_output(double ** d_alpha, double ** d_theta, double * d_periods, curandState * d_states) {
	cudaMalloc(d_alpha, sizeof(double) * h_n_steps * h_n_osc); //TODO cudaMalloc2d?
	cudaMalloc(d_theta, sizeof(double) * h_n_steps * h_n_osc); //TODO cudaMalloc2d?

	//Randomization of the alpha_k(t=0) for each k
	dim3 ra_block(MAX_BLOCK_SIZE);
	dim3 ra_grid((h_n_osc-1) / MAX_BLOCK_SIZE + 1);
	randomize_t0<<<ra_grid, ra_block>>>(*d_alpha, *d_theta, d_periods, d_states);
}

__global__ void randomize_t0(double * d_alpha, double * d_theta, double * d_periods, curandState * d_states) {
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < d_n_osc) {
		d_alpha[idx] = curand_uniform_double(&d_states[idx]) * d_period;
		d_theta[idx] = 2 * PI * d_alpha[idx] / d_periods[idx];
	}
}
