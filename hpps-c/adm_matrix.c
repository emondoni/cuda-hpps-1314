/*
 * adm_matrix.cu
 *
 *  Created on: 02/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "adm_matrix.h"
#include "sim_parameters.h"
#include "samples.h"
#include <stdlib.h>

/*** FUNCTION IMPLEMENTATIONS ***/
void generate_matrix(double ** matrix) {
	double *pattern;
	*matrix = malloc(sizeof(double) * n_osc * n_osc);
	pattern = malloc(sizeof(double) * n_osc);

	init_matrix(*matrix); //initialization of the matrix
	for (int i = 0; i < RANDOM_PASSES; i++) {
		random_pattern(pattern); //generation of a random pattern
		pattern_to_matrix(*matrix, pattern); //generation of the matrix
	}

	finalize_matrix(*matrix);

	free(pattern);
}

void init_matrix(double * matrix) {
	for (unsigned int i = 0; i < n_osc * n_osc; i++)
		matrix[i] = 0.0;
}

void random_pattern(double * pattern) {
	for (unsigned int i = 0; i < n_osc; i++)
		pattern[i] = (rand() & 1) ? 1 : -1;
}

void pattern_to_matrix(double * matrix, double * pattern) {
	for (unsigned int i = 0; i < n_osc; i++)
		for (unsigned int j = 0; j < n_osc; j++)
			if (i != j)
				matrix[j + i * n_osc] = matrix[j + i * n_osc] + pattern[i] * pattern[j];
}

void finalize_matrix(double * matrix) {
	for (unsigned int i = 0; i < n_osc * n_osc; i++)
		matrix[i] = - Y_MAG * matrix[i];
}
