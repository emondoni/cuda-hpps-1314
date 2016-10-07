/*
 * rng.h
 *
 *  Created on: 02/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef RNG_H_
#define RNG_H_

#include <curand_kernel.h>

/*** FUNCTION PROTOTYPES ***/
void initialize_rng(unsigned int n_rng, unsigned long long seed, curandState ** d_states);
__global__ void init_rng(unsigned int n_rng, unsigned long long seed, curandState * d_states);

#endif /* RNG_H_ */
