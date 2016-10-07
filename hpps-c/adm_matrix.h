/*
 * adm_matrix.h
 *
 *  Created on: 02/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef ADM_MATRIX_H_
#define ADM_MATRIX_H_

/*** MACRO DEFINITIONS ***/
#define RANDOM_PASSES 2 // number of randomization passes while building the Y matrix
#define Y_MAG 1e-05 // order of magnitude of the elements of the admittance matrix Y

/*** FUNCTION PROTOTYPES ***/
void generate_matrix(double ** matrix);
void init_matrix(double * matrix);
void random_pattern(double * pattern);
void pattern_to_matrix(double * matrix, double * pattern);
void finalize_matrix(double * matrix);

#endif /* ADM_MATRIX_H_ */
