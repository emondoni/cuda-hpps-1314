/*
 * adm_matrix.h
 *
 *  Created on: 02/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef ADM_MATRIX_H_
#define ADM_MATRIX_H_

#include <curand_kernel.h>

/*** MACRO DEFINITIONS ***/
#define RANDOM_PASSES 2 // number of randomization passes while building the Y matrix
#define Y_MAG 1e-05 // order of magnitude of the elements of the admittance matrix Y

/*** FUNCTION PROTOTYPES ***/
void generate_matrix(double ** d_matrix, curandState * d_states);
__global__ void init_matrix(double * d_matrix);
__global__ void random_pattern(double * d_pattern, curandState * d_states);
__global__ void pattern_to_matrix(double * d_matrix, double * d_pattern);
__global__ void finalize_matrix(double * d_matrix);

#endif /* ADM_MATRIX_H_ */
