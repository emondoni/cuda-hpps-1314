/*
 * env_vars.cu
 *
 *  Created on: 01/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "samples.h"
#include "err.h"
#include <stdio.h>
#include <string.h>
#include <mat.h> //include MATLAB's MAT-file APIs

/*** CONSTANT DEFINITIONS ***/
double *gamma_samp; // experimental samples of Gamma(t)
double *vo_samp; // experimental samples of Vo(t)
double *instants; // instants of time when Gamma and Vo were sampled
unsigned int n_samples; //number of experimental samples for the Gamma and Vo functions

/*** FUNCTION IMPLEMENTATIONS ***/
void allocate_samples(char * samples_filename) {
	MATFile *samples;
	mxArray *gamma, *vo, *insts, *n_samp;

	samples = matOpen(samples_filename, "r");
	if (samples == NULL) {
		fprintf(stderr, "Error while opening the samples file! Did you enter the right path?\n");
		exit(FILE_OPENING_ERROR);
	}

	gamma = matGetVariable(samples, "gamma");
	if (gamma == NULL) {
		fprintf(stderr, "Error while reading gamma samples from the .mat file. Please check its integrity.\n");
		exit(MATFILE_ERROR);
	}

	vo = matGetVariable(samples, "vo");
	if(vo == NULL) {
		fprintf(stderr, "Error while reading vo samples from the .mat file. Please check its integrity.\n");
		exit(MATFILE_ERROR);
	}

	insts = matGetVariable(samples, "instants");
	if (insts == NULL) {
		fprintf(stderr, "Error while reading instant samples from the .mat file. Please check its integrity.\n");
		exit(MATFILE_ERROR);
	}

	n_samp = matGetVariable(samples, "n_samples");
	if (n_samp == NULL) {
		fprintf(stderr, "Error while reading n_samples from the .mat file. Please check its integrity.\n");
		exit(MATFILE_ERROR);
	}

	//Decode n_samp to double and convert it to unsigned int
	n_samples = (unsigned int) mxGetScalar(n_samp);
	if (!(n_samples == mxGetNumberOfElements(gamma) - 1 && n_samples == mxGetNumberOfElements(vo) - 1 && n_samples == mxGetNumberOfElements(insts))) {
		fprintf(stderr, "n_samples doesn't match with the dimensions of gamma, vo and/or instants. Please correct the .mat file for consistency.\n");
		exit(MATFILE_ERROR);
	}

	//Allocate memory for the samples
	gamma_samp = (double *) malloc(sizeof(double) * (n_samples + 1));
	vo_samp = (double *) malloc(sizeof(double) * (n_samples + 1));
	instants = (double *) malloc(sizeof(double) * n_samples);
	if (gamma_samp == NULL || vo_samp == NULL || instants == NULL) {
		fprintf(stderr, "Failed to allocate memory for samples.\n");
		exit(MEM_ALLOCATION_ERROR);
	}

	//Copy sample data to the previously allocated areas
	memcpy((void *)gamma_samp, mxGetPr(gamma), sizeof(double) * (n_samples + 1));
	memcpy((void *)vo_samp, mxGetPr(vo), sizeof(double) * (n_samples + 1));
	memcpy((void *)instants, mxGetPr(insts), sizeof(double) * n_samples);

	matClose(samples);
}
