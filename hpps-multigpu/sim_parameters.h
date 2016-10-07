/*
 * sim_parameters.h
 *
 *  Created on: 04/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef SIM_PARAMETERS_H_
#define SIM_PARAMETERS_H_

/*** MACRO DEFINITIONS ***/
#define PI 3.141592653589793 //pi
#define SIM_FREQ 8 //perform simulation in one instant every SIM_FREQ instants
#define NOISE_ELL -100.0 //noise parameter

/*** GLOBAL VARIABLE DECLARATIONS ***/
//FORMALLY variables (because they have to be assigned at runtime)
//PRACTICALLY constants (they never change during the execution)
extern unsigned int h_n_osc; //defined in multigpu.cu
extern unsigned int h_n_per; //defined in multigpu.cu
extern unsigned int h_noisy; //defined in multigpu.cu
extern double h_period; //defined in init_osc.cu
extern double h_tstop; //defined in init_osc.cu
extern double h_tstep; //defined in init_osc.cu
extern unsigned int h_n_steps; //defined in init_osc.cu
extern double h_omega_0; //defined in init_osc.cu

/*** CONSTANT MEMORY DECLARATIONS ***/
extern __constant__ unsigned int d_n_osc; //defined in multigpu.cu
extern __constant__ unsigned int d_n_per; //defined in multigpu.cu
extern __constant__ double d_period; //defined in init_osc.cu
extern __constant__ double d_tstop; //defined in init_osc.cu
extern __constant__ double d_tstep; //defined in init_osc.cu
extern __constant__ unsigned int d_n_steps; //defined in init_osc.cu
extern __constant__ double d_omega_0; //defined in init_osc.cu


#endif /* SIM_PARAMETERS_H_ */
