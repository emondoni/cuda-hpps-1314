/*
 * output_file.h
 *
 *  Created on: 11/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef OUTPUT_FILE_H_
#define OUTPUT_FILE_H_

#define ALPHA_FILE "alpha.sim"
#define THETA_FILE "theta.sim"
#define TIME_FILE "time.sim"

/*** FUNCTION PROTOTYPES ***/
int matrix_to_file(double * ptr, char * filename, unsigned int rows, unsigned int cols);

#endif /* OUTPUT_FILE_H_ */
