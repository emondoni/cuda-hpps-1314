/*
 * simulate.cu
 *
 *  Created on: 09/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "simulate.h"
#include "samples.h"
#include "sim_parameters.h"
#include "cuda_settings.h"
#include "rng.h"
#include <math.h>
#include <curand_kernel.h>

/*** CONSTANT MEMORY DEFINITIONS ***/
__constant__ double d_sigma_wn;

/*** FUNCTION IMPLEMENTATIONS ***/
void perform_simulation(double * d_alpha, double * d_theta, double * d_time, double * d_matrix,
						double * d_periods, unsigned int noisy_flag, curandState * d_states) {
	double *d_temp_vj;

	dim3 s_block(MAX_BLOCK_SIZE);
	dim3 s_grid((h_n_osc-1) / MAX_BLOCK_SIZE + 1);
	cudaMalloc(&d_temp_vj, sizeof(double) * s_grid.x * s_block.x); //every thread gets a cell in the array
	if (!noisy_flag) {
		simulate<<<s_grid, s_block>>>(d_alpha, d_theta, d_time, d_periods, d_matrix, d_temp_vj);
	} else {
		double seq_w = pow(1e06 * h_period, 2) * pow(10.0, NOISE_ELL / 10.0);
		double h_sigma_wn = sqrt(seq_w / (2 * h_tstep));
		cudaMemcpyToSymbol(d_sigma_wn, &h_sigma_wn, sizeof(double));
		simulate_noisy<<<s_grid, s_block>>>(d_alpha, d_theta, d_time, d_periods, d_matrix, d_temp_vj, d_states);
	}

	//Freeing resources
	cudaFree(d_temp_vj);
}

__global__ void simulate(double * d_alpha, double * d_theta, double * d_time,
						 double * d_periods, double * d_matrix, double * d_temp_vj) {
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	double omega = 2 * PI / d_periods[idx < d_n_osc ? idx : 0];
	double alpha_prev = d_alpha[idx < d_n_osc ? idx : 0];
	double temp_gamma, current;

	/*
	 * The most immediate way to exclude out-of-bounds threads (in the last block), i.e. the if-guard
	 * enclosing the entire code (idx < d_n_osc) is not adequate to the current situation since the code
	 * contains a __syncthreads() instruction. __syncthreads()' behavior is undefined (and very likely
	 * to lead to infinite wait times) if enclosed in an if-clause which is not guaranteed to evaluate
	 * to the same truth value for ALL threads in a block. While that is true for every block up to
	 * the (n-1)-th, the last block likely contains some threads which do not correspond to an oscillator
	 * and which should remain idle.
	 * To solve this issue, we extended the d_temp_vj array to contain a cell for out-of-bounds threads
	 * as well. Those threads will compute meaningless garbage values, get to __syncthreads() and then be
	 * left out of the rest of the code: the last part of the for-loop, in fact, CAN be enclosed in
	 * an if-clause because it doesn't contain synchronization primitives.
	 */
	for (unsigned int t = 1; t < d_n_steps; t++) {
		d_temp_vj[idx] = vo_lut(d_time[t] + alpha_prev, d_periods[idx < d_n_osc ? idx : 0]);
		__syncthreads(); //wait for the temporary values to be available

		if(idx < d_n_osc) {
			temp_gamma = gamma_lut(d_time[t] + alpha_prev, d_periods[idx < d_n_osc ? idx : 0]);
			current = 0.0;
			//This for loop could become another kernel if we could use Dynamic Parallelism
			//(CUDA >= 5.0 and Compute Capability >= 3.5 required) --> ROOM FOR ENHANCEMENTS
			for (unsigned int k = 0; k < d_n_osc; k++)
				current = current + d_matrix[idx + k * d_n_osc] * d_temp_vj[k];

			alpha_prev = alpha_prev + d_tstep * (temp_gamma * current); //calculation of alpha
			d_alpha[idx + t * d_n_osc] = alpha_prev; //storing alpha in the result matrix
			d_theta[idx + t * d_n_osc] = omega * (d_time[t] + alpha_prev); //calculating theta
		}
	}
}

__global__ void simulate_noisy(double * d_alpha, double * d_theta, double * d_time,
							   double * d_periods, double * d_matrix, double * d_temp_vj, curandState * d_states) {
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	double omega = 2 * PI / d_periods[idx < d_n_osc ? idx : 0];
	double alpha_prev = d_alpha[idx < d_n_osc ? idx : 0];
	double temp_gamma, current;

	//See comments to the non-noisy version of the kernel
	for (unsigned int t = 1; t < d_n_steps; t++) {
		d_temp_vj[idx] = vo_lut(d_time[t] + alpha_prev, d_periods[idx < d_n_osc ? idx : 0]);
		__syncthreads(); //wait for the temporary values to be available

		if(idx < d_n_osc) {
			temp_gamma = gamma_lut(d_time[t] + alpha_prev, d_periods[idx < d_n_osc ? idx : 0]);
			current = 0.0;

			for (unsigned int k = 0; k < d_n_osc; k++)
				current = current + d_matrix[idx + k * d_n_osc] * d_temp_vj[k];

			alpha_prev = alpha_prev + d_tstep * (temp_gamma * current + d_sigma_wn * curand_normal_double(&d_states[idx])); //calculation of alpha
			d_alpha[idx + t * d_n_osc] = alpha_prev; //storing alpha in the result matrix
			d_theta[idx + t * d_n_osc] = omega * (d_time[t] + alpha_prev); //calculating theta
		}
	}
}

__device__ double gamma_lut(double instant, double wave_period) {
	double tau = fmod(instant, wave_period);
	double index = N_SAMPLES * tau / wave_period;
	unsigned int int_index = (unsigned int) index;

	return d_gamma[int_index] + (index - int_index) * (d_gamma[int_index < N_SAMPLES ? int_index + 1 : N_SAMPLES] - d_gamma[int_index]);
}

__device__ double vo_lut(double instant, double wave_period) {
	double tau = fmod(instant, wave_period);
	double index = N_SAMPLES * tau / wave_period;
	unsigned int int_index = (unsigned int) index;

	return d_vo[int_index] + (index - int_index) * (d_vo[int_index < N_SAMPLES ? int_index + 1 : N_SAMPLES] - d_vo[int_index]);
}
