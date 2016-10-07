/*
 * output_file.cu
 *
 *  Created on: 11/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "output_file.h"
#include "err.h"
#include "sim_parameters.h"
#include <stdio.h>
#include <string.h>
#include <mat.h>

/*** FUNCTION IMPLEMENTATIONS ***/
int write_mat_output(double * alpha, double * theta, double * time) {
	MATFile *output_file;
	char filename[] = "sim_output.mat";
	int status = 0;

	output_file = matOpen(filename, "w");
	if (output_file == NULL) {
		fprintf(stderr, "Unable to open output file. Do you have the permissions to write in this folder?\n");
		exit(MATFILE_ERROR);
	}

	status += write_array(output_file, "alpha", alpha, n_steps, n_osc);
	status += write_array(output_file, "theta", theta, n_steps, n_osc);
	status += write_array(output_file, "time", time, 1, n_steps);
	status += matClose(output_file);

	if (status) {
		fprintf(stderr, "Errors occurred while writing and/or closing the output file.\n");
		return MATFILE_ERROR;
	}

	return OK;
}

int write_array(MATFile * file, const char * varname, double * array, unsigned int rows, unsigned int cols) {
	mxArray *mat_array;
	int status;

	//NOTE: the row-column inversion in mxCreateDoubleArray is necessary to circumvent
	//the difference with which MATLAB and C treat arrays. C stores arrays in row-major
	//order (i.e. contiguous values belong to the same row), whereas MATLAB does the
	//opposite (i.e. column-major order: contiguous values belong to the same column).
	mat_array = mxCreateDoubleMatrix(cols, rows, mxREAL);
	memcpy((void *)mxGetPr(mat_array), array, sizeof(double) * rows * cols);

	status = matPutVariable(file, varname, mat_array); //assuming file is open
	mxDestroyArray(mat_array);
	if (status) {
		fprintf(stderr, "Error while writing variable %s to file.\n", varname);
		return MATFILE_ERROR;
	}

	return OK;
}
