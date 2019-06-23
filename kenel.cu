/*
 * kenel.cu
 *
 *  Created on: May 19, 2019
 *      Author: Andrés Pupiales Arévalo
 *      apupiales@gmail.com
 *      https://github.com/apupiales
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * The GNU General Public License is available at:
 *   http://www.gnu.org/copyleft/gpl.html
 * or by writing to:
 *     The Free Software Foundation, Inc.,
 *     675 Mass Ave, Cambridge, MA 02139, USA.
 */

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

//Custom sources
#include "settings_KC10_2fl_1uni.cu"
//#include "settings_KC10_2fl_1uni.cu"
//#include "settings_KC10_2fl_2uni.cu"
//#include "settings_KC10_2fl_3uni.cu"
//#include "settings_KC20_2fl_1uni.cu"
//#include "settings_KC20_2fl_2uni.cu"
//#include "settings_KC20_2fl_3uni.cu"
//#include "settings_KC30_2fl_1uni.cu"
//#include "settings_KC30_2fl_2uni.cu"

__global__ void curand_setup(curandState *state, int seed){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}


/**
 * This function shuffles chromosomes genes randomly over all population
*/
__global__ void shuffle_population_genes(curandState *my_curandstate, const unsigned *max_rand_int, const unsigned *min_rand_int, int population[POPULATION_SIZE][FACILITIES_LOCATIONS]) {

	int i = blockIdx.x;
	if (i < POPULATION_SIZE) {
		for (int j = 0; j < FACILITIES_LOCATIONS; j++) {
			int idx = j + blockDim.x * blockIdx.x;

			float myrandf = curand_uniform(my_curandstate + idx);
			int myrand = int(myrandf * 10);

			if (myrand != population[i][j]) {
				int current_value = population[i][j];
				population[i][j] = population[i][myrand];
				population[i][myrand] = current_value;
			}
		}
	}
}

/**
 * This function creates the base population with POPULATION_SIZE chromosomes
 */
 __global__ void generateBasePopulation(int population[POPULATION_SIZE][FACILITIES_LOCATIONS]){
      int i = blockIdx.x;
      int j = threadIdx.x;
      if (i < POPULATION_SIZE && j < FACILITIES_LOCATIONS) {
    	  population[i][j] = j;
      }
  }

int main() {
	/* To measure the execution time */
	clock_t begin = clock();
	/* To set seed variable */
	time_t t;
	/* To check correct synchronization */
	cudaError_t cudaStatus;

	/* Initializes random number generator */
	srand((unsigned) time(&t));

	/* seed for curand */
	int seed = rand() % 10000;

	/* Variable for population in host memory */
	int *h_population[POPULATION_SIZE][FACILITIES_LOCATIONS];

	/* Variable for population in device memory */
	int *d_population;
	cudaMalloc((void**)&d_population, sizeof(int) * POPULATION_SIZE * FACILITIES_LOCATIONS);

	/* Generation of all base chromosomes (genes ordered ascending) */
	generateBasePopulation<<<POPULATION_SIZE, 32>>>((int (*)[FACILITIES_LOCATIONS]) d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError!");
	}
	/* Set population in host memory from device memory */
	cudaMemcpy(h_population, d_population, POPULATION_SIZE*(FACILITIES_LOCATIONS)*sizeof(int), cudaMemcpyDeviceToHost);

	/* Uncommet this section of code to print the base population
	printf("Base Population\n");
	for (int i = 0; i < POPULATION_SIZE; i++) {
		printf("Chromosome %d\n", i);
		for (int j = 0; j < FACILITIES_LOCATIONS; j++) {
			printf("%d ", h_population[i][j]);
		}
		printf("\n");
	}
	*/

	/* Initialize variables for random values generation with curand */
	curandState *d_state;
	cudaMalloc(&d_state, sizeof(curandState));
	unsigned *d_max_rand_int, *h_max_rand_int, *d_min_rand_int, *h_min_rand_int;
	cudaMalloc(&d_max_rand_int, sizeof(unsigned));
	h_max_rand_int = (unsigned *) malloc(sizeof(unsigned));
	cudaMalloc(&d_min_rand_int, sizeof(unsigned));
	h_min_rand_int = (unsigned *) malloc(sizeof(unsigned));
	*h_max_rand_int = FACILITIES_LOCATIONS;
	*h_min_rand_int = 0;
	cudaMemcpy(d_max_rand_int, h_max_rand_int, sizeof(unsigned),cudaMemcpyHostToDevice);
	cudaMemcpy(d_min_rand_int, h_min_rand_int, sizeof(unsigned),cudaMemcpyHostToDevice);

	curand_setup<<<POPULATION_SIZE, 32>>>(d_state, seed);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError!");
	}
	/* Shuffles chromosome genes randomly over all population */
	shuffle_population_genes<<<POPULATION_SIZE, 1>>>(d_state, d_max_rand_int,
			d_min_rand_int, (int (*)[FACILITIES_LOCATIONS]) d_population);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError!");
	}

	/* Set current population (with Shuffled genes) in host memory from device memory */
	 cudaMemcpy(h_population, d_population, POPULATION_SIZE*(FACILITIES_LOCATIONS)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError!");
	}

	/* Uncommet this section of code to print the base population
	printf("Shuffled Population\n");
	for (int i = 0; i < POPULATION_SIZE; i++) {
		printf("Chromosome %d\n", i);
		for (int j = 0; j < FACILITIES_LOCATIONS; j++) {
			printf("%d ", h_population[i][j]);
		}
		printf("\n");
	}
	*/

	clock_t end = clock();
	double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
	printf("\n Time Spent: %f", time_spent);

	exit(0);
}

