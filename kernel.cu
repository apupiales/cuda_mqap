/*
 * kernel.cu
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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cufft.h"

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

// In NSGA2, the Rt population size.
const int NSGA2_POPULATION_SIZE = POPULATION_SIZE * 2;

__global__ void curand_setup(curandState* state, int seed) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

/**
 * This function creates the base population with POPULATION_SIZE chromosomes
 */
__global__ void generateBasePopulation(short population[][FACILITIES_LOCATIONS]) {

	if (threadIdx.x < FACILITIES_LOCATIONS) {
		population[blockIdx.x][threadIdx.x] = threadIdx.x;
	}
}

/**
 * This function shuffles chromosomes genes randomly over all population
 */
__global__ void shufflePopulationGenes(curandState* my_curandstate,
	const unsigned* max_rand_int, const unsigned* min_rand_int,
	short population[][FACILITIES_LOCATIONS]) {

#pragma unroll
	for (int j = 0; j < FACILITIES_LOCATIONS; j++) {
		int idx = j + blockDim.x * blockIdx.x;

		float myrandf = curand_uniform(my_curandstate + idx);
		int myrand = int(myrandf * 10);

		if (myrand != population[blockIdx.x][j]) {
			short current_value = population[blockIdx.x][j];
			population[blockIdx.x][j] = population[blockIdx.x][myrand];
			population[blockIdx.x][myrand] = current_value;
		}
	}
}

/**
 * This function convert the base population to its binary 2d matrix representation.
 * population_2d variable will have the original binary 2d matrix representation
 * of each chromosome and population_2d_transposed will have the transposed version
 * of each binary 2d matrix representation.
 */
__global__ void populationTo2DRepresentation(
	short population[][FACILITIES_LOCATIONS],
	short population_2d[][FACILITIES_LOCATIONS],
	short population_2d_transposed[][FACILITIES_LOCATIONS]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;

	if (j < FACILITIES_LOCATIONS && k < FACILITIES_LOCATIONS) {
		population_2d[j + (i * FACILITIES_LOCATIONS)][k] = 0;
		population_2d_transposed[k + (i * FACILITIES_LOCATIONS)][j] = 0;
		if (population[i][j] == k) {
			population_2d[j + (i * FACILITIES_LOCATIONS)][k] = 1;
			population_2d_transposed[k + (i * FACILITIES_LOCATIONS)][j] = 1;
		}

	}
}

/**
 *  Multiplication between selected flow matrix and input_matrix
 *  (in this strict order)
 */
__global__ void multiplicationWithFlowMatrix(int flow_matrix_id,
	short input_matrix[][FACILITIES_LOCATIONS],
	unsigned int output_matrix[][FACILITIES_LOCATIONS]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;

	if (j < FACILITIES_LOCATIONS && k < FACILITIES_LOCATIONS) {
		unsigned int sum = 0;
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
	unsigned int input_matrix[][FACILITIES_LOCATIONS],
	unsigned int output_matrix[][FACILITIES_LOCATIONS]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;

	if (j < FACILITIES_LOCATIONS && k < FACILITIES_LOCATIONS) {
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
__global__ void matrixMultiplication(unsigned int input_matrix_a[][FACILITIES_LOCATIONS],
	short input_matrix_b[][FACILITIES_LOCATIONS],
	unsigned int output_matrix[][FACILITIES_LOCATIONS]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;

	if (j < FACILITIES_LOCATIONS && k < FACILITIES_LOCATIONS) {
		int sum = 0;
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			sum += input_matrix_a[j + (i * FACILITIES_LOCATIONS)][x]
				* input_matrix_b[x + (i * FACILITIES_LOCATIONS)][k];
		}
		output_matrix[j + (i * FACILITIES_LOCATIONS)][k] = sum;
	}
}

__global__ void calculateTrace(short objective_id,
	unsigned int input_matrix[][FACILITIES_LOCATIONS],
	unsigned int d_population_fitness[][OBJECTIVES]) {

	int i = blockIdx.x;
	unsigned int sum = 0;
	for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
		sum += input_matrix[x + (i * FACILITIES_LOCATIONS)][x];
	}
	d_population_fitness[i][objective_id] = sum;
}

/**
 * This function calculates the fitness of each objective for all chromosomes in the 
 * population. The flow matrix is multiplied with the chromosome (represented in a 
 * binary 2d matrix), the resultant matrix is multiplied with the distance transposed
 * matrix, then the resultant matrix is multiplied with the transposed chromosome
 * (also a binary 2d matrix). The trace of this resultant matrix is the chromosome's
 * fitness. The fitness must be calculated for each flow matrix.
 * Trace(Fn*X*DT*XT)
 */
void calculatePopulationFitness(
	short h_population[][FACILITIES_LOCATIONS], unsigned int h_population_fitness[][OBJECTIVES], 
	short d_population[][FACILITIES_LOCATIONS], unsigned int d_population_fitness[][OBJECTIVES]) {

	/* Variable to check correct synchronization */
	cudaError_t cudaStatus;

	/*******************************************************************************************
	 * Comment this section if you don't need to print partial results of fitness calculation.
	 */

	// Variable for population binary 2d representation in host memory (X)
	short h_2d_population[NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable for population binary 2d representation transposed in host memory (XT)
	short h_2d_transposed_population[NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep Fn*X result in host memory (Fn: Flow matrix n)
	int h_temporal_1[NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep Fn*X*DT result in host memory (DT: Transposed Distances matrix)
	int h_temporal_2[NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep Fn*X*DT*XT result in host memory
	int h_temporal_3[NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	/********************************************************************************************/

	/* Variable for population binary 2d representation in device memory (X) */
	short(*d_2d_population)[FACILITIES_LOCATIONS];
	cudaMalloc((void**)&d_2d_population,
		sizeof(short) * NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS
		* FACILITIES_LOCATIONS);

	/* Variable for population binary 2d representation transposed in device memory (XT) */
	short(*d_2d_transposed_population)[FACILITIES_LOCATIONS];
	cudaMalloc((void**)&d_2d_transposed_population,
		sizeof(short) * NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS
		* FACILITIES_LOCATIONS);

	/*
	 * Variable to keep Fn*X result in device memory (Fn: Flow matrix n).
	 * This variable is also used to keep Fn*X*DT*XT result
	 */
	unsigned int(*d_temporal_1)[FACILITIES_LOCATIONS];
	cudaMalloc((void**)&d_temporal_1,
		sizeof(unsigned int) * NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS
		* FACILITIES_LOCATIONS);

	/* Variable to keep Fn*X*DT result in device memory (DT: Transposed Distances matrix) */
	unsigned int(*d_temporal_2)[FACILITIES_LOCATIONS];
	cudaMalloc((void**)&d_temporal_2,
		sizeof(unsigned int) * NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS
		* FACILITIES_LOCATIONS);
	
	for (int obj = 0; obj < OBJECTIVES; obj++) {

		dim3 threads(32, 32);
		populationTo2DRepresentation <<<NSGA2_POPULATION_SIZE, threads >>> (d_population,
			d_2d_population, d_2d_transposed_population);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "populationTo2DRepresentation Sync CudaError!\n");
		}

		/*********************************************************************
		 * Comment this section if you don't need to print the partial results
		 * of fitness calculation.
		 */

		// Set current population binary 2d representation in host memory from device memory
		cudaMemcpy(h_2d_population, d_2d_population,
		NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(short),
				cudaMemcpyDeviceToHost);
		// Set current population binary 2d representation transposed in host memory from device memory
		cudaMemcpy(h_2d_transposed_population, d_2d_transposed_population,
		NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(short),
				cudaMemcpyDeviceToHost);
		/*********************************************************************/

		/*
		 * Fn*X
		 */
		multiplicationWithFlowMatrix <<<NSGA2_POPULATION_SIZE, threads >>> (obj,
			d_2d_population, d_temporal_1);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "multiplicationWithFlowMatrix Sync CudaError!\n");
		}

		/*********************************************************************
		 * Comment this section if you don't need to print the partial results
		 * of fitness calculation.
		 */

		// Set the result of F1*X in host memory from device memory
		cudaMemcpy(h_temporal_1, d_temporal_1,
			NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS
						* sizeof(unsigned int), cudaMemcpyDeviceToHost);
		/*********************************************************************/

		/*
		 * Fn*X*DT
		 */
		multiplicationWithTranposedDistanceMatrix <<<NSGA2_POPULATION_SIZE, threads >>> (
			d_temporal_1, d_temporal_2);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "multiplicationWithTranposedDistanceMatrix Sync CudaError!\n");
		}

		/*********************************************************************
		 * Comment this section if you don't need to print the partial results
		 * of fitness calculation.
		 */

		// Set the result of Fn*X*DT in host memory from device memory
		cudaMemcpy(h_temporal_2, d_temporal_2,
			NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(unsigned int),
				cudaMemcpyDeviceToHost);
		/*********************************************************************/
		
		/*
		 * Fn*X*DT*XT
		 */
		matrixMultiplication <<<NSGA2_POPULATION_SIZE, threads >>> (d_temporal_2,
			d_2d_transposed_population, d_temporal_1);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "matrixMultiplication Sync CudaError!\n");
		}
		/*********************************************************************
		 * Comment this section if you don't need to print the partial results
		 * of fitness calculation.
		 */

		// Set the result of Fn*X*DT*XT in host memory from device memory
		cudaMemcpy(h_temporal_3, d_temporal_1,
			NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(unsigned int),
				cudaMemcpyDeviceToHost);
		/*********************************************************************/

		/*
		 * Trace(Fn*X*DT*XT)
		 */
		calculateTrace <<<NSGA2_POPULATION_SIZE, 1 >>> (obj, d_temporal_1, d_population_fitness);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "calculateTrace 1 Sync CudaError!\n");
		}

		/* Set current population fitness in host memory from device memory */
		cudaMemcpy(h_population_fitness, d_population_fitness,
			NSGA2_POPULATION_SIZE * OBJECTIVES * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);

		/*********************************************************************
		 * Comment this section if you don't need to print the partial results
		 * of fitness calculation.
		 * NOTE: If you uncomment this section, all previous sections with this
		 * notes MUST be uncomented too.
		 */
		printf("\nPopulation IN FITNESS OPERATION\n");
		for (int i = 0; i < NSGA2_POPULATION_SIZE; i++) {
			printf("Chromosome %d\n", i);
			for (int j = 0; j < FACILITIES_LOCATIONS; j++) {
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

			printf("\n F%d*X \n", obj);
			for (int x = 0; x < FACILITIES_LOCATIONS; x++) {

				for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
					printf("%d ", h_temporal_1[x + (i * FACILITIES_LOCATIONS)][y]);
				}
				printf("\n");
			}

			printf("\n F%d*X*DT \n", obj);
			for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
				for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
					printf("%d ", h_temporal_2[x + (i * FACILITIES_LOCATIONS)][y]);
				}
				printf("\n");
			}

			printf("\n F%d*X*DT*XT \n", obj);
			for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
				for (int y = 0; y < FACILITIES_LOCATIONS; y++) {
					printf("%d ", h_temporal_3[x + (i * FACILITIES_LOCATIONS)][y]);
				}
				printf("\n");
			}
			printf("\n");
		}
		/*********************************************************************/
	}

	cudaFree(d_2d_population);
	cudaFree(d_2d_transposed_population);
	cudaFree(d_temporal_1);
	cudaFree(d_temporal_2);
	
}


int main()
{
	// In NSGA2, the Rt population size.
	//const int NSGA2_POPULATION_SIZE = POPULATION_SIZE * 2;
	// To measure the execution time.
	clock_t begin = clock();
	// To set seed variable.
	time_t t;
	// To check correct synchronization.
	cudaError_t cudaStatus;
	// Initializes random number generator.
	srand((unsigned)time(&t));
	// Seed for curand.
	int seed = rand() % 10000;

	/***********VARIABLES IN HOST MEMORY*********/
	// Variable for population  in host memory.
	short h_population[NSGA2_POPULATION_SIZE][FACILITIES_LOCATIONS];
	// Variable for population fitness in host memory.
	unsigned int h_population_fitness[NSGA2_POPULATION_SIZE][OBJECTIVES];
	// Variable for population rank in host memory.
	short h_population_rank[NSGA2_POPULATION_SIZE];
	// Variable for population crowding distance in host memory.
	float h_population_crowding[NSGA2_POPULATION_SIZE];

	/**********VARIABLES IN DEVICE MEMORY********/
	// Variable for population in device memory.
	short(*d_population)[FACILITIES_LOCATIONS];
	cudaMalloc((void**)&d_population, sizeof(short) * NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS);
	// Variable for population fitness in device memory.
	unsigned int(*d_population_fitness)[OBJECTIVES];
	cudaMalloc((void**)&d_population_fitness, sizeof(unsigned int) * FACILITIES_LOCATIONS);
	// Variable for population rank in device memory.
	short(*d_population_rank);
	cudaMalloc((void**)&d_population_rank, sizeof(short) * NSGA2_POPULATION_SIZE);
	// Variable for population crowding distance in device memory.
	float(*d_population_crowding);
	cudaMalloc((void**)&d_population_crowding, sizeof(float) * NSGA2_POPULATION_SIZE);

	/* Generation of all base chromosomes (genes ordered ascending).
	 * 64 threads are defined here because we are going to tackle instances upto 60 FACILITIES/LOCATIONS.
	 */
	generateBasePopulation <<<NSGA2_POPULATION_SIZE, 64 >>> (d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "generateBasePopulation Sync CudaError!\n");
	}

	/* Uncomment this section of code to print the base population
	 Set population in host memory from device memory */
	cudaMemcpy(h_population, d_population,
		NSGA2_POPULATION_SIZE * (FACILITIES_LOCATIONS) * sizeof(short),
		cudaMemcpyDeviceToHost);
	printf("\nBase Population\n");
	for (int i = 0; i < NSGA2_POPULATION_SIZE; i++) {
		printf("Chromosome %d\n", i);
		for (int j = 0; j < FACILITIES_LOCATIONS; j++) {
			printf("%d ", h_population[i][j]);
		}
		printf("\n");
	}
	/* */

	 /* Initialize variables for random values generation with curand */
	curandState* d_state;
	cudaMalloc(&d_state, sizeof(curandState));
	unsigned* d_max_rand_int, * h_max_rand_int, * d_min_rand_int, * h_min_rand_int;
	cudaMalloc(&d_max_rand_int, sizeof(unsigned));
	h_max_rand_int = (unsigned*)malloc(sizeof(unsigned));
	cudaMalloc(&d_min_rand_int, sizeof(unsigned));
	h_min_rand_int = (unsigned*)malloc(sizeof(unsigned));
	*h_max_rand_int = FACILITIES_LOCATIONS;
	*h_min_rand_int = 0;
	cudaMemcpy(d_max_rand_int, h_max_rand_int, sizeof(unsigned),
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_min_rand_int, h_min_rand_int, sizeof(unsigned),
		cudaMemcpyHostToDevice);

	// 64 threads are defined here because we are going to tackle instances upto 60 FACILITIES/LOCATIONS.
	curand_setup <<<NSGA2_POPULATION_SIZE, 64 >>> (d_state, seed);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "curand_setup Sync CudaError!");
	}

	/* Shuffles chromosome genes randomly over all population */
	shufflePopulationGenes <<<NSGA2_POPULATION_SIZE, 1 >>> (d_state, d_max_rand_int,
		d_min_rand_int, d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "shufflePopulationGenes Sync CudaError!");
	}
	/* Uncomment this section of code to print the Shuffled population
	 Set current population (with Shuffled genes) in host memory from device memory */
	cudaMemcpy(h_population, d_population,
		NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS * sizeof(short),
		cudaMemcpyDeviceToHost);

	printf("\nShuffled Population\n");
	for (int i = 0; i < NSGA2_POPULATION_SIZE; i++) {
		printf("Chromosome %d\n", i);
		for (int j = 0; j < FACILITIES_LOCATIONS; j++) {
			printf("%d ", h_population[i][j]);
		}
		printf("\n");
	}
	/**/


	/* Set all chromosomes with 1 2 7 9 6 5 0 4 3 8 0 0 for test purposes 
	 * expected fitness: F0 = 228322 F1 = 193446
	 */
	
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
	}

	// Set Initial population in device memory from host memory with a fixed solution
	cudaMemcpy(d_population, h_population,
			NSGA2_POPULATION_SIZE * (FACILITIES_LOCATIONS) * sizeof(short),
				cudaMemcpyHostToDevice);

	printf("\nInitial Population\n");
	for (int i = 0; i < NSGA2_POPULATION_SIZE; i++) {
		printf("Chromosome %d\n", i);
		// Print solution.
		for (int j = 0; j < FACILITIES_LOCATIONS; j++) {
			printf("%d ", h_population[i][j]);
		}
		// Print fitness.
		for (int j = 0; j < OBJECTIVES; j++) {
			printf("|%d ", h_population_fitness[i][j]);
		}
		// Print rank.
		printf("|%d ", h_population_rank[i]);
		// Print crowding.
		printf("%f", h_population_crowding[i]);
		printf("\n");
	}
	
	/******************************************************************/

	for (int iteration = 1; iteration <= ITERATIONS; iteration++) {

		/* Calculate fitness on each population chromosome */
		calculatePopulationFitness(h_population, h_population_fitness, d_population, d_population_fitness);

	}
	printf("\nfitness\n");
	for (int i = 0; i < NSGA2_POPULATION_SIZE; i++) {
		for (int j = 0; j < OBJECTIVES; j++) {
			printf("|%d ", h_population_fitness[i][j]);
		}
		printf("\n");
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;


	/* Uncomment this section of code to print the Shuffled population
	Set current population (with Shuffled genes) in host memory from device memory */
	cudaMemcpy(h_population, d_population,
		NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS * sizeof(short),
		cudaMemcpyDeviceToHost);
	cudaMemcpy(h_population_fitness, d_population_fitness,
		NSGA2_POPULATION_SIZE * OBJECTIVES * sizeof(unsigned int),
		cudaMemcpyDeviceToHost);
	cudaMemcpy(h_population_rank, d_population_rank,
		NSGA2_POPULATION_SIZE * sizeof(short),
		cudaMemcpyDeviceToHost);
	cudaMemcpy(h_population_crowding, d_population_crowding,
		NSGA2_POPULATION_SIZE * sizeof(float),
		cudaMemcpyDeviceToHost);

	printf("\nPopulation with fitness\n");
	for (int i = 0; i < NSGA2_POPULATION_SIZE; i++) {
		printf("Chromosome %d\n", i);
		// Print solution.
		for (int j = 0; j < FACILITIES_LOCATIONS; j++) {
			printf("%d ", h_population[i][j]);
		}
		// Print fitness.
		for (int j = 0; j < OBJECTIVES; j++) {
			printf("|%d ", h_population_fitness[i][j]);
		}
		// Print rank.
		printf("|%d ", h_population_rank[i]);
		// Print crowding.
		printf("%f", h_population_crowding[i]);
		printf("\n");
	}
	/* */

	printf("\n Time Spent: %f", time_spent);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

