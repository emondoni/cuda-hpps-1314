/*
 * samples.h
 *
 *  Created on: 01/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef SAMPLES_H_
#define SAMPLES_H_

/*** MACRO DEFINITIONS ***/
#define N_SAMPLES 200 //number of experimental samples for the Gamma and Vo functions

/*** CONSTANT DECLARATIONS ***/
extern const double h_gamma[]; //experimental samples of Gamma(t)
extern const double h_vo[]; //experimental samples of Vo(t)
extern const double h_instants[]; //instants of time when Gamma and Vo were sampled

/*** CONSTANT MEMORY DECLARATIONS ***/
extern __device__ double d_gamma[N_SAMPLES + 1];
extern __device__ double d_vo[N_SAMPLES + 1];
extern __device__ double d_instants[N_SAMPLES];

/*** FUNCTION PROTOTYPES ***/
void allocate_samples(const unsigned int version);

#endif /* SAMPLES_H_ */
