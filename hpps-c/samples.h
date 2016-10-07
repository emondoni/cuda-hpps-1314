/*
 * samples.h
 *
 *  Created on: 01/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef SAMPLES_H_
#define SAMPLES_H_

/*** CONSTANT DECLARATIONS ***/
extern double *gamma_samp; //experimental samples of Gamma(t)
extern double *vo_samp; //experimental samples of Vo(t)
extern double *instants; //instants of time when Gamma and Vo were sampled
extern unsigned int n_samples; //number of experimental samples for the Gamma and Vo functions

/*** FUNCTION PROTOTYPES ***/
void allocate_samples(char * samples_filename);

#endif /* SAMPLES_H_ */
