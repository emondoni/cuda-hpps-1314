/*
 * init_output.cu
 *
 *  Created on: 06/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "init_output.h"
#include "sim_parameters.h"
#include <stdlib.h>

void initialize_output(double ** alpha, double ** theta) {
	*alpha = malloc(sizeof(double) * n_steps * n_osc);
	*theta = malloc(sizeof(double) * n_steps * n_osc);

	//Randomization of the alpha_k(t=0) for each k
	randomize_t0(*alpha, *theta);
}

void randomize_t0(double * alpha, double * theta) {
	for (unsigned int i = 0; i < n_osc; i++) {
		alpha[i] = (((double) rand()) / RAND_MAX) * period;
		theta[i] = 2 * PI * alpha[i] / period;
	}
}
