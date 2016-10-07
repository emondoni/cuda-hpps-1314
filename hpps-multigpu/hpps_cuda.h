/*
 * multigpu.h
 *
 *  Created on: 02/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef HPPS_CUDA_H_
#define HPPS_CUDA_H_

#define DEVICE_RES 1
#define HOST_RES 2

/*** FUNCTION PROTOTYPES ***/
void process_arguments(int argc, char** argv);
void free_resources(int n_params, ...);
#endif /* HPPS_CUDA_H_ */
