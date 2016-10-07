/*
 * simulate.h
 *
 *  Created on: 09/dic/2014
 *  Author: Edoardo Mondoni
 */

#ifndef SIMULATE_H_
#define SIMULATE_H_

/*** FUNCTION PROTOTYPES ***/
void perform_simulation(double * alpha, double * theta, double * time, double * matrix, double * periods, unsigned int noisy_flag);
void simulate(double * alpha, double * theta, double * time, double * matrix, double * periods);
void simulate_noisy(double * alpha, double * theta, double * time, double * matrix, double * periods);
double gamma_lut(double instant, double wave_period);
double vo_lut(double instant, double wave_period);
double rand_normal();

#endif /* SIMULATE_H_ */
