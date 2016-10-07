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
#include "init_osc.h"
#include "init_output.h"
#include "simulate.h"
#include "output_file.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "hpps_c.h"

/*** GLOBAL VARIABLE DEFINITIONS ***/
unsigned int n_osc;
unsigned int n_per;
unsigned int noisy;

int main(int argc, char** argv) {
	char *samples_filename;
	double *matrix, *periods, *times, *alpha, *theta;

	/* PRELIMINARY PHASE
	 * The arguments are validated and stored in the appropriate variables, including the samples
	 * of the Gamma(t) and Vo(t) functions along with the time grid used for their sampling.
	 * The program initializes the oscillators' properties and the simulation based on the constant
	 * parameters #defined in macros and those derivable from the Gamma(t) and Vo(t) functions.
	 * Then, the transadmittance matrix (d_matrix) is randomly generated and stored in device memory
	 * for later use.
	 */
	process_arguments(argc, argv, &samples_filename);
	srand(time(NULL));
	allocate_samples(samples_filename); // copies Gamma(t) and Vo(t) samples (and instants) into dev
	initialize_simulation(&periods, &times);
	generate_matrix(&matrix);

	/* OUTPUT INITIALIZATION PHASE
	 * The d_alpha and d_theta pointers are initialized (i.e. memory is allocated which will contain
	 * the results of the simulation). Furthermore, the alpha_k(0) are stored in the appropriate
	 * memory locations (they can be generated here since they are randomized).
	 */
	initialize_output(&alpha, &theta);

	/* SIMULATION PHASE
	 * This is the core of the program: it is where the simulation happens.
	 */
	perform_simulation(alpha, theta, times, matrix, periods, noisy);

	/* OUTPUT COPY PHASE
	 * The results of the simulation are copied to a .mat file.
	 */
	write_mat_output(alpha, theta, times);

	/* RESOURCE FREEING PHASE
	 * Given that the results of the simulation have been written to files, we can free
	 * the memory we allocated dynamically before exiting.
	 */
	free_resources(8, matrix, periods, times, alpha, theta, gamma_samp, vo_samp, instants);
}

void process_arguments(int argc, char** argv, char ** samples_filename) {
	if (argc < 5) {
		fprintf(stderr, "Usage: hpps-c <n_oscillators> <n_periods> <noisy_flag> <samples_filename>\n");
		exit(NOT_ENOUGH_PARAMS);
	}

	int ret1 = sscanf(argv[1], "%u", &n_osc); //store the number of oscillators into the host variable
	int ret2 = sscanf(argv[2], "%u", &n_per); //store the number of periods into the host variable
	int ret3 = sscanf(argv[3], "%u", &noisy); //store the noisy flag into the host variable
	*samples_filename = argv[4]; //store the .mat file's path's address in samples_filename

	if (ret1 != 1 || ret2 != 1 || ret3 != 1 || *samples_filename == NULL) {
		fprintf(stderr, "An error has occurred while parsing the parameters. Please check their consistency.\n");
		exit(WRONG_PARAMS);
	} else if (n_osc < 1) {
		fprintf(stderr, "Invalid number of oscillators. Valid values: > 1\n");
		exit(INVALID_N_OSC);
	} else if (n_per < 1) {
		fprintf(stderr, "Invalid number of periods. Valid values: > 1\n");
		exit(INVALID_N_PER);
	} else if (!(noisy == 0 || noisy == 1)) {
		fprintf(stderr, "Invalid noisy flag. Valid values: 0 (no noise), 1 (noise)\n");
		exit(INVALID_NOISY_FLAG);
	}
}

void free_resources(int n_resources, ...) {
	if (n_resources < 1)
		return;

	va_list param_list;
	va_start(param_list, n_resources);

	for (unsigned int i = 1; i <= n_resources; i++)
		free(va_arg(param_list, double*));

	va_end(param_list);
}

