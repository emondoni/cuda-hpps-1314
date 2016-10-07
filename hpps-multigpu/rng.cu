/*
 * rng.cu
 *
 *  Created on: 02/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "rng.h"
#include "sim_parameters.h"
#include "cuda_settings.h"
#include <curand_kernel.h>

/*** FUNCTION IMPLEMENTATIONS ***/
void initialize_rng(unsigned int n_rng, unsigned long long seed, curandState ** d_states) {
	cudaMalloc(d_states, sizeof(curandState) * n_rng); //allocation of device memory for the states
	dim3 rng_block(MAX_BLOCK_SIZE); //definition of block size
	dim3 rng_grid((n_rng-1) / MAX_BLOCK_SIZE + 1); //definition of grid size
	init_rng<<<rng_grid, rng_block>>>(n_rng, seed, *d_states); //kernel call
}

__global__ void init_rng(unsigned int n_rng, unsigned long long seed, curandState * d_states) {
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n_rng)
		curand_init(seed, idx, 0, &d_states[idx]); //actual initialization of the RNGs
}
