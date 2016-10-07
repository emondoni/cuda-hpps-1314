/*
 * init_osc.cu
 *
 *  Created on: 04/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "init_osc.h"
#include "sim_parameters.h"
#include "samples.h"
#include "rng.h"
#include "cuda_settings.h"
#include <curand_kernel.h>

/*** CONSTANT DEFINITIONS ***/
double h_period;
double h_tstop;
double h_tstep;
unsigned int h_n_steps;
double h_omega_0;

/*** CONSTANT MEMORY DEFINITIONS ***/
__constant__ double d_period;
__constant__ double d_tstop;
__constant__ double d_tstep;
__constant__ unsigned int d_n_steps;
__constant__ double d_omega_0;

/*** FUNCTION DEFINITIONS ***/
void initialize_simulation(double ** d_periods, double ** d_time, curandState * d_states) {
	//Determining the period from the time grid
	h_period = h_instants[N_SAMPLES - 1];
	cudaMemcpyToSymbol(d_period, &h_period, sizeof(double));

	//Period randomization
	cudaMalloc(d_periods, sizeof(double) * h_n_osc);
	dim3 rp_block(MAX_BLOCK_SIZE);
	dim3 rp_grid((h_n_osc-1) / MAX_BLOCK_SIZE + 1);
	randomize_periods<<<rp_grid, rp_block>>>(*d_periods, d_states);

	//Calculation of other parameters of interest and copy to constant memory
	h_tstop = h_period * h_n_per; //derive the simulation time interval
	h_tstep = SIM_FREQ * (h_instants[1] - h_instants[0]); //derive the simulation timestep
	h_n_steps = (int)(h_tstop / h_tstep) + 1; //add 1 to include t0 = 0;
	h_omega_0 = 2 * PI / h_period; //determine the *free-running* pulsation of the oscillators
	cudaMemcpyToSymbol(d_tstop, &h_tstop, sizeof(double));
	cudaMemcpyToSymbol(d_tstep, &h_tstep, sizeof(double));
	cudaMemcpyToSymbol(d_n_steps, &h_n_steps, sizeof(unsigned int));
	cudaMemcpyToSymbol(d_omega_0, &h_omega_0, sizeof(double));

	//Generation of the simulation time vector
	cudaMalloc(d_time, sizeof(double) * h_n_steps);
	dim3 gtv_block(MAX_BLOCK_SIZE);
	dim3 gtv_grid((h_n_steps-1) / MAX_BLOCK_SIZE + 1);
	generate_time_vector<<<gtv_grid, gtv_block>>>(*d_time);
}

__global__ void randomize_periods(double * d_periods, curandState * d_states) {
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < d_n_osc)
		d_periods[idx] = d_period * (1 + RAND_COEF * (curand_uniform_double(&d_states[idx]) - 0.5));
}

__global__ void generate_time_vector(double * d_time) {
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < d_n_steps)
		d_time[idx] = d_tstep * idx;
}
