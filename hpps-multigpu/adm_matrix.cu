/*
 * adm_matrix.cu
 *
 *  Created on: 02/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "adm_matrix.h"
#include "sim_parameters.h"
#include "samples.h"
#include "rng.h"
#include "cuda_settings.h"
#include <curand_kernel.h>

/*** FUNCTION IMPLEMENTATIONS ***/
void generate_matrix(double ** d_matrix, curandState * d_states) {
	double *d_pattern;
	cudaMalloc(d_matrix, sizeof(double) * h_n_osc * h_n_osc); //TODO cudaMalloc2d?
	cudaMalloc(&d_pattern, sizeof(double) * h_n_osc);

	dim3 init_block(MAX_BLOCK_SIZE); //block size for init_matrix
	dim3 init_grid((h_n_osc*h_n_osc-1) / MAX_BLOCK_SIZE + 1); //grid size for init_matrix
	dim3 rp_block(MAX_BLOCK_SIZE); //block size for random_pattern
	dim3 rp_grid((h_n_osc-1) / MAX_BLOCK_SIZE + 1); //grid size for random_pattern
	dim3 ptm_block(IDEAL_BLOCK_X, IDEAL_BLOCK_Y); //block size for pattern_to_matrix
	dim3 ptm_grid((h_n_osc-1) / IDEAL_BLOCK_X + 1, (h_n_osc-1) / IDEAL_BLOCK_Y + 1); //grid size for pattern_to_matrix
	unsigned int ptm_shmem = sizeof(double) * (ptm_block.x + ptm_block.y);

	init_matrix<<<init_grid, init_block>>>(*d_matrix); //initialization of the matrix
	for (int i = 0; i < RANDOM_PASSES; i++) {
		random_pattern<<<rp_grid, rp_block>>>(d_pattern, d_states); //generation of a random pattern
		pattern_to_matrix<<<ptm_grid, ptm_block, ptm_shmem>>>(*d_matrix, d_pattern); //generation of the matrix
	}

	dim3 final_block = init_block;
	dim3 final_grid = init_grid;
	finalize_matrix<<<final_grid, final_block>>>(*d_matrix);

	cudaFree(d_pattern);
}

__global__ void init_matrix(double * d_matrix) {
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < d_n_osc * d_n_osc)
		d_matrix[idx] = 0.0;
}

__global__ void random_pattern(double * d_pattern, curandState * d_states) {
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < d_n_osc) {
		unsigned int x = curand(&d_states[idx]); //generating a random integer
		d_pattern[idx] = (x & 1) ? 1 : -1; //setting the pattern cell to 1 or -1 based on x's oddity
	}
}

__global__ void pattern_to_matrix(double * d_matrix, double * d_pattern) {
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

	extern __shared__ double shmem[]; //shared memory for the local copy of the relevant portions of d_pattern!
	const unsigned int offset = blockDim.x; //how far into the shmem array begins the "vertical" portion
	if (x < d_n_osc && y < d_n_osc) {
		if (threadIdx.y == 0)
			shmem[threadIdx.x] = d_pattern[x]; //fill the "horizontal" portion
		if (threadIdx.x == 0)
			shmem[offset + threadIdx.y] = d_pattern[y]; //fill the "vertical" portion
	}

	__syncthreads(); //wait for shmem to be appropriately and completely filled

	if (x < d_n_osc && y < d_n_osc && x != y)
		d_matrix[x + y * d_n_osc] = d_matrix[x + y * d_n_osc] + shmem[threadIdx.x] * shmem[offset + threadIdx.y];
}

__global__ void finalize_matrix(double * d_matrix) {
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < d_n_osc * d_n_osc)
		d_matrix[idx] = - Y_MAG * d_matrix[idx];
}
