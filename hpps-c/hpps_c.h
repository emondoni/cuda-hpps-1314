/*
 * multigpu.h
 *
 *  Created on: 02/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef HPPS_C_H_
#define HPPS_C_H_

#define RNG_SEED 17021991

/*** FUNCTION PROTOTYPES ***/
void process_arguments(int argc, char** argv, char ** samples_filename);
void free_resources(int n_params, ...);
#endif /* HPPS_C_H_ */
