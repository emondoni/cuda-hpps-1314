/*
 * output_file.h
 *
 *  Created on: 11/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef OUTPUT_FILE_H_
#define OUTPUT_FILE_H_

#include <mat.h>

/*** FUNCTION PROTOTYPES ***/
int write_mat_output(double * h_alpha, double * h_theta, double * h_time);
int write_array(MATFile * file, const char * varname, double * array, unsigned int rows, unsigned int cols);

#endif /* OUTPUT_FILE_H_ */
