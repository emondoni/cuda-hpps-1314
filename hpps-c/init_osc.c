/*
 * init_osc.cu
 *
 *  Created on: 04/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "init_osc.h"
#include "sim_parameters.h"
#include "samples.h"
#include <stdlib.h>

/*** CONSTANT DEFINITIONS ***/
double period;
double tstop;
double tstep;
unsigned int n_steps;
double omega_0;

/*** FUNCTION DEFINITIONS ***/
void initialize_simulation(double ** periods, double ** time) {
	//Determining the period from the time grid
	period = instants[n_samples - 1];

	//Period randomization
	*periods = malloc(sizeof(double) * n_osc);
	randomize_periods(*periods);

	//Calculation of other parameters of interest and copy to constant memory
	tstop = period * n_per; //derive the simulation time interval
	tstep = SIM_FREQ * (instants[1] - instants[0]); //derive the simulation timestep
	n_steps = (int)(tstop / tstep) + 1; //add 1 to include t0 = 0;
	omega_0 = 2 * PI / period; //determine the *free-running* pulsation of the oscillators

	//Generation of the simulation time vector
	*time = malloc(sizeof(double) * n_steps);
	generate_time_vector(*time);
}

void randomize_periods(double * periods) {
	for (unsigned int i = 0; i < n_osc; i++)
		periods[i] = period * (1 + RAND_COEF * (((double) rand()) / RAND_MAX - 0.5));
}

void generate_time_vector(double * time) {
	for (unsigned int i = 0; i < n_steps; i++)
		time[i] = tstep * i;
}
