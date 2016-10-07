/*
 * simulate.cu
 *
 *  Created on: 09/dic/2014
 *  Author: Edoardo Mondoni
 */

#include "simulate.h"
#include "samples.h"
#include "sim_parameters.h"
#include <math.h>
#include <stdlib.h>

/*** FUNCTION IMPLEMENTATIONS ***/
void perform_simulation(double * alpha, double * theta, double * time, double * matrix,
						double * periods, unsigned int noisy_flag) {
	if (!noisy_flag)
		simulate(alpha, theta, time, matrix, periods);
	else
		simulate_noisy(alpha, theta, time, matrix, periods);
}

void simulate(double * alpha, double * theta, double * time, double * matrix, double * periods) {
	double omega[n_osc], current, *alpha_prev, *alpha_cur, *theta_cur;

	alpha_prev = alpha;
	alpha_cur = alpha + n_osc;
	theta_cur = theta + n_osc;
	for (unsigned int i = 0; i < n_osc; i++)
		omega[i] = 2 * PI / periods[i];

	for (unsigned int t = 1; t < n_steps; t++) {
		for (unsigned int k = 0; k < n_osc; k++) {
			current = 0.0;

			for (unsigned int kk = 0; kk < n_osc; kk++)
				current = current + matrix[kk + k * n_osc] * vo_lut(time[t] + alpha_prev[kk], periods[kk]);

			alpha_cur[k] = alpha_prev[k] + tstep * gamma_lut(time[t] + alpha_prev[k], periods[k]) * current;
			theta_cur[k]= omega[k] * (time[t] + alpha_cur[k]);
		}

		alpha_prev = alpha_cur;
		alpha_cur = alpha_cur + n_osc;
		theta_cur = theta_cur + n_osc;
	}
}

void simulate_noisy(double * alpha, double * theta, double * time, double * matrix, double * periods) {
	double sigma_wn = sqrt((pow(1e06 * period, 2) * pow(10.0, NOISE_ELL / 10.0)) / (2 * tstep));
	double omega[n_osc], current, *alpha_prev, *alpha_cur, *theta_cur;

	alpha_prev = alpha;
	alpha_cur = alpha + n_osc;
	theta_cur = theta + n_osc;
	for (unsigned int i = 0; i < n_osc; i++)
		omega[i] = 2 * PI * periods[i];

	for (unsigned int t = 1; t < n_steps; t++) {
		for (unsigned int k = 0; k < n_osc; k++) {
			current = 0.0;

			for (unsigned int kk = 0; kk < n_osc; kk++)
				current = current + matrix[kk + k * n_osc] * vo_lut(time[t] + alpha_prev[kk], periods[kk]);

			alpha_cur[k] = alpha_prev[k] + tstep * (gamma_lut(time[t] + alpha_prev[k], periods[k]) * current + sigma_wn * rand_normal());
			theta_cur[k]= omega[k] * (time[t] + alpha_cur[k]);
		}

		alpha_prev = alpha_cur;
		alpha_cur = alpha_cur + n_osc;
		theta_cur = theta_cur + n_osc;
	}
}

double gamma_lut(double instant, double wave_period) {
	double tau = fmod(instant, wave_period);
	double index = n_samples * tau / wave_period;
	unsigned int int_index = index >= 0 ? (unsigned int) index : (unsigned int) -index;

	return gamma_samp[int_index] + (index - int_index) * (gamma_samp[int_index < n_samples ? int_index + 1 : n_samples] - gamma_samp[int_index]);
}

double vo_lut(double instant, double wave_period) {
	double tau = fmod(instant, wave_period);
	double index = n_samples * tau / wave_period;
	unsigned int int_index = index >= 0 ? (unsigned int) index : (unsigned int) -index;

	return vo_samp[int_index] + (index - int_index) * (vo_samp[int_index < n_samples ? int_index + 1 : n_samples] - vo_samp[int_index]);
}

double rand_normal()
{
	static double U, V, Z;
	static int phase = 0;

	if(phase == 0) {
		U = (rand() + 1.0) / (RAND_MAX + 2.0);
		V = rand() / (RAND_MAX + 1.0);
		Z = sqrt(-2 * log(U)) * sin(2 * PI * V);
	} else
		Z = sqrt(-2 * log(U)) * cos(2 * PI * V);

	phase = 1 - phase;

	return Z;
}
