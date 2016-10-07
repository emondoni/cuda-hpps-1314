/*
 * output_file.cu
 *
 *  Created on: 11/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "output_file.h"
#include "err.h"
#include <stdio.h>

/*** FUNCTION IMPLEMENTATIONS ***/
int matrix_to_file(double * ptr, char * filename, unsigned int rows, unsigned int cols) {
	FILE *fp;
	fp = fopen(filename, "w+");

	if (fp == NULL) {
		fprintf(stderr, "Error while opening output file.");
		return FILE_OPENING_ERROR;
	}

	for (unsigned int i = 0; i < rows; i++) {
		for(unsigned int j = 0; j < cols - 1; j++)
			fprintf(fp, "%.12g, ", ptr[j + cols * i]);
		fprintf(fp, "%.12g\n", ptr[cols * (i+1) - 1]);
	}

	fclose(fp);
	return OK;
}
