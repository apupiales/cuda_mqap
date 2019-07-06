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

__global__ void curand_setup(curandState *state, int seed) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

/**
 * This function creates the base population with POPULATION_SIZE chromosomes
 */
__global__ void generateBasePopulation(
		int population[][FACILITIES_LOCATIONS + OBJECTIVES]) {

	int i = blockIdx.x;
	int j = threadIdx.x;

	if (i < POPULATION_SIZE && j < FACILITIES_LOCATIONS + OBJECTIVES) {
		if (j < FACILITIES_LOCATIONS) {
			population[i][j] = j;
		} else {
			/* This positions will be use to allocate the fitness value
			 * for each objective and is initialized with 0 */
			population[i][j] = 0;
		}
	}
}

/**
 * This function shuffles chromosomes genes randomly over all population
 */
__global__ void shufflePopulationGenes(curandState *my_curandstate,
		const unsigned *max_rand_int, const unsigned *min_rand_int,
		int population[][FACILITIES_LOCATIONS + OBJECTIVES]) {

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
 * This function creates a base population binary 2d matrix representation.
 * population_2d variable will have the original binary 2d matrix representation
 * of each chromosome and population_2d_transposed will have the transposed version
 * of each binary 2d matrix representation.
 */
__global__ void populationTo2DRepresentation(
		int population[][FACILITIES_LOCATIONS + OBJECTIVES],
		int population_2d[][FACILITIES_LOCATIONS],
		int population_2d_transposed[][FACILITIES_LOCATIONS]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;

	if (i < POPULATION_SIZE && j < FACILITIES_LOCATIONS
			&& k < FACILITIES_LOCATIONS) {
		if (population[i][j] == k) {
			population_2d[j + (i * FACILITIES_LOCATIONS)][k] = 1;
			population_2d_transposed[k + (i * FACILITIES_LOCATIONS)][j] = 1;
		} else {
			population_2d[j + (i * FACILITIES_LOCATIONS)][k] = 0;
			population_2d_transposed[k + (i * FACILITIES_LOCATIONS)][j] = 0;
		}
	}
}

/**
 *  Multiplication between selected flow matrix and input_matrix
 *  (in this strict order)
 */
__global__ void multiplicationWithFlowMatrix(int flow_matrix_id,
		int input_matrix[][FACILITIES_LOCATIONS],
		int output_matrix[][FACILITIES_LOCATIONS]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;

	if (i < POPULATION_SIZE && j < FACILITIES_LOCATIONS
			&& k < FACILITIES_LOCATIONS) {
		int sum = 0;
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			sum += d_flowMatrices[flow_matrix_id][j][x]
					* input_matrix[x + (i * FACILITIES_LOCATIONS)][k];
		}
		output_matrix[j + (i * FACILITIES_LOCATIONS)][k] = sum;
	}
}

/**
 *  Multiplication between Transposed distance matrix and input_matrix
 *  (in this strict order)
 */
__global__ void multiplicationWithTranposedDistanceMatrix(
		int input_matrix[][FACILITIES_LOCATIONS],
		int output_matrix[][FACILITIES_LOCATIONS]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;

	if (i < POPULATION_SIZE && j < FACILITIES_LOCATIONS
			&& k < FACILITIES_LOCATIONS) {
		int sum = 0;

		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			sum += input_matrix[j + (i * FACILITIES_LOCATIONS)][x]
					* d_transposeDistancesMatrix[x][k];
		}
		output_matrix[j + (i * FACILITIES_LOCATIONS)][k] = sum;
	}
}

/**
 *  Multiplication between matrix a and matrix b
 */
__global__ void matrixMultiplication(int input_matrix_a[][FACILITIES_LOCATIONS],
		int input_matrix_b[][FACILITIES_LOCATIONS],
		int output_matrix[][FACILITIES_LOCATIONS]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;

	if (i < POPULATION_SIZE && j < FACILITIES_LOCATIONS
			&& k < FACILITIES_LOCATIONS) {
		int sum = 0;
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			sum += input_matrix_a[j + (i * FACILITIES_LOCATIONS)][x]
					* input_matrix_b[x + (i * FACILITIES_LOCATIONS)][k];
		}
		output_matrix[j + (i * FACILITIES_LOCATIONS)][k] = sum;
	}
}

__global__ void calculateTrace(int objective_id,
		int input_matrix[][FACILITIES_LOCATIONS],
		int population[][FACILITIES_LOCATIONS + OBJECTIVES]) {

	int i = blockIdx.x;

	if (i < POPULATION_SIZE) {
		int sum = 0;
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			sum += input_matrix[x + (i * FACILITIES_LOCATIONS)][x];
		}
		population[i][FACILITIES_LOCATIONS + objective_id] = sum;
	}
}

/**
 * This function calculates the fitness of all chromosomes in the population.
 * The flow matrix is multiplied with the chromosome (represented in a binary
 * 2d matrix), the resultant matrix is multiplied with the distance transposed
 * matrix, then the resultant matrix is multiplied with the transposed chromosome
 * (also a binary 2d matrix). The trace of this resultant matrix is the chromosome's
 * fitness. The fitness must be calculated for each flow matrix.
 * Trace(Fn*X*DT*XT)
 */
void calculatePopulationfitness(
		int h_population[][FACILITIES_LOCATIONS + OBJECTIVES], int d_population[][FACILITIES_LOCATIONS + OBJECTIVES]) {

	/* Variable to check correct synchronization */
	cudaError_t cudaStatus;

	/*******************************************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 */

	/* Variable for population binary 2d representation in host memory (X)*/
	int h_2d_population[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	/* Variable for population binary 2d representation transposed in host memory (XT) */
	int h_2d_transposed_population[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	/* Variable to keep F1*X result in host memory (F1: Flow matrix 1)*/
	int h_temporal_1[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	/* Variable to keep F2*X result in host memory (F2: Flow matrix 2)*/
	int h_temporal_2[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	/* Variable to keep F1*X*DT result in host memory (DT: Transposed Distances matrix) */
	int h_temporal_3[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	/* Variable to keep F2*X*DT result in host memory (DT: Transposed Distances matrix) */
	int h_temporal_4[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	/* Variable to keep F1*X*DT*XT result in host memory*/
	int h_temporal_5[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	/* Variable to keep F2*X*DT*XT result in host memory*/
	int h_temporal_6[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	/********************************************************************************************/

	/* Variable for population binary 2d representation in device memory (X)*/
	int (*d_2d_population)[FACILITIES_LOCATIONS];
	cudaMalloc((void**) &d_2d_population,
			sizeof(int) * POPULATION_SIZE * FACILITIES_LOCATIONS
					* FACILITIES_LOCATIONS);

	/* Variable for population binary 2d representation transposed in device memory (XT) */
	int (*d_2d_transposed_population)[FACILITIES_LOCATIONS];
	cudaMalloc((void**) &d_2d_transposed_population,
			sizeof(int) * POPULATION_SIZE * FACILITIES_LOCATIONS
					* FACILITIES_LOCATIONS);

	/*
	 * Variable to keep F1*X result in device memory (F1: Flow matrix 1).
	 * This variable is also use to keep F1*X*DT*XT result
	 */
	int (*d_temporal_1)[FACILITIES_LOCATIONS];
	cudaMalloc((void**) &d_temporal_1,
			sizeof(int) * POPULATION_SIZE * FACILITIES_LOCATIONS
					* FACILITIES_LOCATIONS);
	/*
	 * Variable to keep F1*X result in device memory (F2: Flow matrix 2).
	 * This variable is also use to keep F2*X*DT*XT result
	 */
	int (*d_temporal_2)[FACILITIES_LOCATIONS];
	cudaMalloc((void**) &d_temporal_2,
			sizeof(int) * POPULATION_SIZE * FACILITIES_LOCATIONS
					* FACILITIES_LOCATIONS);
	/* Variable to keep F1*X*DT result in device memory (DT: Transposed Distances matrix) */
	int (*d_temporal_3)[FACILITIES_LOCATIONS];
	cudaMalloc((void**) &d_temporal_3,
			sizeof(int) * POPULATION_SIZE * FACILITIES_LOCATIONS
					* FACILITIES_LOCATIONS);
	/* Variable to keep F2*X*DT result in device memory (DT: Transposed Distances matrix) */
	int (*d_temporal_4)[FACILITIES_LOCATIONS];
	cudaMalloc((void**) &d_temporal_4,
			sizeof(int) * POPULATION_SIZE * FACILITIES_LOCATIONS
					* FACILITIES_LOCATIONS);

	dim3 threads(32, 32);
	populationTo2DRepresentation<<<POPULATION_SIZE, threads>>>(d_population,
			d_2d_population, d_2d_transposed_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError!");
	}

	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 */

	/* Set current population binary 2d representation in host memory from device memory */
	cudaMemcpy(h_2d_population, d_2d_population,
	POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(int),
			cudaMemcpyDeviceToHost);
	/* Set current population binary 2d representation transposed in host memory from device memory */
	cudaMemcpy(h_2d_transposed_population, d_2d_transposed_population,
	POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(int),
			cudaMemcpyDeviceToHost);
	/*********************************************************************/

	/*
	 * F1*X
	 */
	multiplicationWithFlowMatrix<<<POPULATION_SIZE, threads>>>(0,
			d_2d_population, d_temporal_1);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError M1!");
	}
	/*
	 * F2*X
	 */
	multiplicationWithFlowMatrix<<<POPULATION_SIZE, threads>>>(1,
			d_2d_population, d_temporal_2);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError M2!");
	}

	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 */

	/* Set the result of F1*X in host memory from device memory */
	cudaMemcpy(h_temporal_1, d_temporal_1,
			POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS
					* sizeof(int), cudaMemcpyDeviceToHost);
	/* Set the result of F2*X in host memory from device memory */
	cudaMemcpy(h_temporal_2, d_temporal_2,
			POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS
					* sizeof(int), cudaMemcpyDeviceToHost);
	/*********************************************************************/

	/*
	 * F1*X*DT
	 */
	multiplicationWithTranposedDistanceMatrix<<<POPULATION_SIZE, threads>>>(
			d_temporal_1, d_temporal_3);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError M3!");
	}
	/*
	 * F2*X*DT
	 */
	multiplicationWithTranposedDistanceMatrix<<<POPULATION_SIZE, threads>>>(
			d_temporal_2, d_temporal_4);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError M4!");
	}

	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 */

	/* Set the result of F1*X*DT in host memory from device memory */
	cudaMemcpy(h_temporal_3, d_temporal_3,
	POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(int),
			cudaMemcpyDeviceToHost);
	/* Set the result of F2*X*DT in host memory from device memory */
	cudaMemcpy(h_temporal_4, d_temporal_4,
	POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(int),
			cudaMemcpyDeviceToHost);
	/*********************************************************************/

	/*
	 * F1*X*DT*XT
	 */
	matrixMultiplication<<<POPULATION_SIZE, threads>>>(d_temporal_3,
			d_2d_transposed_population, d_temporal_1);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError M5!");
	}
	/*
	 * F2*X*DT*XT
	 */
	matrixMultiplication<<<POPULATION_SIZE, threads>>>(d_temporal_4,
			d_2d_transposed_population, d_temporal_2);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError M6!");
	}

	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 */

	/* Set the result of F1*X*DT*XT in host memory from device memory */
	cudaMemcpy(h_temporal_5, d_temporal_1,
	POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(int),
			cudaMemcpyDeviceToHost);
	/* Set the result of F2*X*DT*XT in host memory from device memory */
	cudaMemcpy(h_temporal_6, d_temporal_1,
	POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(int),
			cudaMemcpyDeviceToHost);
	/*********************************************************************/

	/*
	 * Trace(F1*X*DT*XT)
	 */
	calculateTrace<<<POPULATION_SIZE, 1>>>(0, d_temporal_1, d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError M7!");
	}
	/*
	 * Trace(F2*X*DT*XT)
	 */
	calculateTrace<<<POPULATION_SIZE, 1>>>(1, d_temporal_2, d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError M8!");
	}

	/* Set current population (with Shuffled genes) in host memory from device memory */
	cudaMemcpy(h_population, d_population,
	POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES) * sizeof(int),
			cudaMemcpyDeviceToHost);


	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 * NOTE: If you uncomment this section, all previous sections with this
	 * notes must be uncomented too.
	 */
	printf("\nPopulation\n");
	for (int i = 0; i < POPULATION_SIZE; i++) {
		printf("Chromosome %d\n", i);
		for (int j = 0; j < FACILITIES_LOCATIONS + OBJECTIVES; j++) {
			printf("%d ", h_population[i][j]);
		}
		printf("\n");

		printf("\n2d Matrix Representation (X) \n");
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {

			for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
				printf("%d ", h_2d_population[x + (i * FACILITIES_LOCATIONS)][y]);
			}
			printf("\n");
		}

		printf("\n2d Matrix Representation (Transposed) (XT)\n");
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {

			for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
				printf("%d ", h_2d_transposed_population[x + (i * FACILITIES_LOCATIONS)][y]);
			}
			printf("\n");
		}

		printf("\n F1*X \n");
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {

			for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
				printf("%d ", h_temporal_1[x + (i * FACILITIES_LOCATIONS)][y]);
			}
			printf("\n");
		}

		printf("\n F1*X*DT \n");
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
				printf("%d ", h_temporal_3[x + (i * FACILITIES_LOCATIONS)][y]);
			}
			printf("\n");
		}

		printf("\n F1*X*DT*XT \n");
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
				printf("%d ", h_temporal_5[x + (i * FACILITIES_LOCATIONS)][y]);
			}
			printf("\n");
		}

		printf("\n F2*X \n");
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
				printf("%d ", h_temporal_2[x + (i * FACILITIES_LOCATIONS)][y]);
			}
			printf("\n");
		}

		printf("\n F2*X*DT \n");
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
				printf("%d ", h_temporal_4[x + (i * FACILITIES_LOCATIONS)][y]);
			}
			printf("\n");
		}

		printf("\n F2*X*DT*XT \n");
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
				printf("%d ", h_temporal_6[x + (i * FACILITIES_LOCATIONS)][y]);
			}
			printf("\n");
		}
		printf("\n");
	}
	/*********************************************************************/

	cudaFree(d_2d_population);
	cudaFree(d_2d_transposed_population);
	cudaFree(d_temporal_1);
	cudaFree(d_temporal_2);
	cudaFree(d_temporal_3);
	cudaFree(d_temporal_4);
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
	int h_population[POPULATION_SIZE][FACILITIES_LOCATIONS + OBJECTIVES];

	/* Variable for population in device memory */
	int (*d_population)[FACILITIES_LOCATIONS + OBJECTIVES];
	cudaMalloc((void**) &d_population,
			sizeof(int) * POPULATION_SIZE
					* (FACILITIES_LOCATIONS + OBJECTIVES));

	/* Generation of all base chromosomes (genes ordered ascending) */
	generateBasePopulation<<<POPULATION_SIZE, 32>>>(d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError!");
	}
	/* Set population in host memory from device memory */
	cudaMemcpy(h_population, d_population,
	POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES) * sizeof(int),
			cudaMemcpyDeviceToHost);

	/* Uncommet this section of code to print the base population
	 printf("\nBase Population\n");
	 for (int i = 0; i < POPULATION_SIZE; i++) {
	 printf("Chromosome %d\n", i);
	 for (int j = 0; j < FACILITIES_LOCATIONS + OBJECTIVES; j++) {
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
	cudaMemcpy(d_max_rand_int, h_max_rand_int, sizeof(unsigned),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_min_rand_int, h_min_rand_int, sizeof(unsigned),
			cudaMemcpyHostToDevice);

	curand_setup<<<POPULATION_SIZE, 32>>>(d_state, seed);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError!");
	}

	/* Shuffles chromosome genes randomly over all population */
	shufflePopulationGenes<<<POPULATION_SIZE, 1>>>(d_state, d_max_rand_int,
			d_min_rand_int, d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sync CudaError!");
	}

	/* Set current population (with Shuffled genes) in host memory from device memory */
	cudaMemcpy(h_population, d_population,
	POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES) * sizeof(int),
			cudaMemcpyDeviceToHost);

	/* Uncommet this section of code to print the Shuffled population
	 printf("\nShuffled Population\n");
	 for (int i = 0; i < POPULATION_SIZE; i++) {
	 printf("Chromosome %d\n", i);
	 for (int j = 0; j < FACILITIES_LOCATIONS + OBJECTIVES; j++) {
	 printf("%d ", h_population[i][j]);
	 }
	 printf("\n");
	 }
	 */

	/* Set all chromosomes with 1 2 7 9 6 5 0 4 3 8 0 0 for test purposes*/
	for (int a = 0; a < POPULATION_SIZE; a++) {
		h_population[a][0] = 1;
		h_population[a][1] = 2;
		h_population[a][2] = 7;
		h_population[a][3] = 9;
		h_population[a][4] = 6;
		h_population[a][5] = 5;
		h_population[a][6] = 0;
		h_population[a][7] = 4;
		h_population[a][8] = 3;
		h_population[a][9] = 8;
		h_population[a][10] = 0;
		h_population[a][11] = 0;
	}

	cudaMemcpy(d_population, h_population,
			POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES) * sizeof(int),
				cudaMemcpyHostToDevice);

	/* Calculate fitness on each population chromosome */
	calculatePopulationfitness(h_population, d_population);

	clock_t end = clock();
	double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
	printf("\n Time Spent: %f", time_spent);

	exit(0);
}

