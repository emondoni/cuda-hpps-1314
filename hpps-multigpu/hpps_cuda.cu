/*
 * multigpu.cu
 *
 *  Created on: 01/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "samples.h"
#include "sim_parameters.h"
#include "adm_matrix.h"
#include "err.h"
#include "multigpu.h"
#include "init_osc.h"
#include "init_output.h"
#include "simulate.h"
#include "output_file.h"
#include "rng.h"
#include <curand_kernel.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

/*** GLOBAL VARIABLE DEFINITIONS ***/
unsigned int h_n_osc;
unsigned int h_n_per;
unsigned int h_noisy;

/*** CONSTANT MEMORY DEFINITIONS ***/
__constant__ unsigned int d_n_osc;
__constant__ unsigned int d_n_per;

int main(int argc, char** argv) {
	double *d_matrix, *d_periods, *d_time, *d_alpha, *d_theta;
	curandState *d_states;

	/* PRELIMINARY PHASE
	 * The arguments are validated and stored in the appropriate variables, including the samples
	 * of the Gamma(t) and Vo(t) functions along with the time grid used for their sampling.
	 * The program initializes the oscillators' properties and the simulation based on the constant
	 * parameters #defined in macros and those derivable from the Gamma(t) and Vo(t) functions.
	 * Then, the transadmittance matrix (d_matrix) is randomly generated and stored in device memory
	 * for later use.
	 */
	process_arguments(argc, argv);
	initialize_rng(h_n_osc, 17021991, &d_states);
	allocate_samples(0); // copies Gamma(t) and Vo(t) samples (and instants) into dev
	initialize_simulation(&d_periods, &d_time, d_states);
	generate_matrix(&d_matrix, d_states);

	/* OUTPUT INITIALIZATION PHASE
	 * The d_alpha and d_theta pointers are initialized (i.e. memory is allocated which will contain
	 * the results of the simulation). Furthermore, the alpha_k(0) are stored in the appropriate
	 * memory locations (they can be generated here since they are randomized).
	 */
	initialize_output(&d_alpha, &d_theta, d_periods, d_states);

	/* SIMULATION PHASE
	 * This is the core of the program: it is where the simulation happens.
	 */
	perform_simulation(d_alpha, d_theta, d_time, d_matrix, d_periods, h_noisy, d_states);

	/* OUTPUT COPY PHASE
	 * The results of the simulation are copied back to host memory and then to a file.
	 */
	double *h_alpha, *h_theta, *h_time;
	h_alpha = (double *) malloc(sizeof(double) * h_n_osc * h_n_steps);
	h_theta = (double *) malloc(sizeof(double) * h_n_osc * h_n_steps);
	h_time = (double *) malloc(sizeof(double) * h_n_steps);
	cudaMemcpy(h_alpha, d_alpha, sizeof(double) * h_n_osc * h_n_steps, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_theta, d_theta, sizeof(double) * h_n_osc * h_n_steps, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_time, d_time, sizeof(double) * h_n_steps, cudaMemcpyDeviceToHost);
	matrix_to_file(h_alpha, ALPHA_FILE, h_n_steps, h_n_osc);
	matrix_to_file(h_theta, THETA_FILE, h_n_steps, h_n_osc);
	matrix_to_file(h_time, TIME_FILE, 1, h_n_steps);

	/* RESOURCE FREEING PHASE
	 *
	 */
	cudaFree(d_states);
	free_resources(8, DEVICE_RES, d_matrix, DEVICE_RES, d_periods, DEVICE_RES, d_time, DEVICE_RES, d_alpha,
					DEVICE_RES, d_theta, HOST_RES, h_alpha, HOST_RES, h_theta, HOST_RES, h_time);
}

void process_arguments(int argc, char** argv) {
	if (argc < 4) {
		fprintf(stderr, "Usage: multigpu <n_oscillators> <n_periods> <noisy_flag>\n");
		exit(NOT_ENOUGH_PARAMS);
	}

	int ret1 = sscanf(argv[1], "%u", &h_n_osc); //store the number of oscillators into the host variable
	int ret2 = sscanf(argv[2], "%u", &h_n_per); //store the number of periods into the host variable
	int ret3 = sscanf(argv[3], "%u", &h_noisy); //store the noisy flag into the host variable

	if (ret1 != 1 || ret2 != 1 || ret3 != 1) {
		fprintf(stderr, "An error has occurred while parsing the parameters. Please check their consistency.\n");
		exit(WRONG_PARAMS);
	} else if (h_n_osc < 1) {
		fprintf(stderr, "Invalid number of oscillators. Valid values: > 1\n");
		exit(INVALID_N_OSC);
	} else if (h_n_per < 1) {
		fprintf(stderr, "Invalid number of periods. Valid values: > 1\n");
		exit(INVALID_N_PER);
	} else if (!(h_noisy == 0 || h_noisy == 1)) {
		fprintf(stderr, "Invalid noisy flag. Valid values: 0 (no noise), 1 (noise)\n");
		exit(INVALID_NOISY_FLAG);
	}

	cudaMemcpyToSymbol(d_n_osc, &h_n_osc, sizeof(unsigned int)); //copy into device memory
	cudaMemcpyToSymbol(d_n_per, &h_n_per, sizeof(unsigned int)); //copy into device memory
}

void free_resources(int n_resources, ...) {
	if (n_resources < 1)
		return;

	va_list param_list;
	va_start(param_list, n_resources);

	for (unsigned int i = 1; i <= n_resources; i++) {
		int res_type = va_arg(param_list, int);

		if (res_type == DEVICE_RES)
			cudaFree(va_arg(param_list, double*));
		else if (res_type == HOST_RES)
			free(va_arg(param_list, double*));
		else {
			fprintf(stderr, "Invalid free_resoures parameters pattern.\n");
			break;
		}
	}

	va_end(param_list);
}

