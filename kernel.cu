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
#include <stdbool.h>
#include <math.h>

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
		#pragma unroll
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
		#pragma unroll
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
		#pragma unroll
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			sum += input_matrix_a[j + (i * FACILITIES_LOCATIONS)][x]
				* input_matrix_b[x + (i * FACILITIES_LOCATIONS)][k];
		}
		output_matrix[j + (i * FACILITIES_LOCATIONS)][k] = sum;
	}
}

__global__ void calculateTrace(short objective_id,
	unsigned int input_matrix[][FACILITIES_LOCATIONS],
	unsigned int d_population_fitness[][OBJECTIVES + 1],
	unsigned int d_sorted_population_fitness[][OBJECTIVES + 1]) {

	int i = blockIdx.x;
	unsigned int sum = 0;
	#pragma unroll
	for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
		sum += input_matrix[x + (i * FACILITIES_LOCATIONS)][x];
	}
	d_population_fitness[i][objective_id] = d_sorted_population_fitness[i][objective_id] = sum;
}

__global__ void setOriginalIndex(unsigned int d_population_fitness[][OBJECTIVES + 1], unsigned int d_sorted_population_fitness[][OBJECTIVES + 1]) {

	d_population_fitness[blockIdx.x][OBJECTIVES] = d_sorted_population_fitness[blockIdx.x][OBJECTIVES] = blockIdx.x;
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
void parallelPopulationFitnessCalculation(
	short h_population[][FACILITIES_LOCATIONS], unsigned int h_population_fitness[][OBJECTIVES + 1],
	short d_population[][FACILITIES_LOCATIONS], unsigned int d_population_fitness[][OBJECTIVES + 1], unsigned int d_sorted_population_fitness[][OBJECTIVES + 1]) {

	/* Variable to check correct synchronization */
	cudaError_t cudaStatus;

	/*******************************************************************************************
	 * Comment this section if you don't need to print partial results of fitness calculation.
	 */

	// Variable for population binary 2d representation in host memory (X).
	short h_2d_population[NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable for population binary 2d representation transposed in host memory (XT).
	short h_2d_transposed_population[NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep Fn*X result in host memory (Fn: Flow matrix n).
	int h_temporal_1[NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep Fn*X*DT result in host memory (DT: Transposed Distances matrix).
	int h_temporal_2[NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep Fn*X*DT*XT result in host memory.
	int h_temporal_3[NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	/********************************************************************************************/

	// Variable for population binary 2d representation in device memory (X).
	short(*d_2d_population)[FACILITIES_LOCATIONS];
	cudaMalloc((void**)&d_2d_population,
		sizeof(short) * NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS
		* FACILITIES_LOCATIONS);

	// Variable for population binary 2d representation transposed in device memory (XT).
	short(*d_2d_transposed_population)[FACILITIES_LOCATIONS];
	cudaMalloc((void**)&d_2d_transposed_population,
		sizeof(short) * NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS
		* FACILITIES_LOCATIONS);

	/*
	 * Variable to keep Fn*X result in device memory (Fn: Flow matrix n).
	 * This variable is also used to keep Fn*X*DT*XT result.
	 */
	unsigned int(*d_temporal_1)[FACILITIES_LOCATIONS];
	cudaMalloc((void**)&d_temporal_1,
		sizeof(unsigned int) * NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS
		* FACILITIES_LOCATIONS);

	// Variable to keep Fn*X*DT result in device memory (DT: Transposed Distances matrix).
	unsigned int(*d_temporal_2)[FACILITIES_LOCATIONS];
	cudaMalloc((void**)&d_temporal_2,
		sizeof(unsigned int) * NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS
		* FACILITIES_LOCATIONS);
	
	for (int obj = 0; obj < OBJECTIVES; obj++) {

		dim3 threads(32, 32);
		populationTo2DRepresentation <<<NSGA2_POPULATION_SIZE, threads>>> (d_population,
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
		multiplicationWithFlowMatrix <<<NSGA2_POPULATION_SIZE, threads>>> (obj,
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
		multiplicationWithTranposedDistanceMatrix <<<NSGA2_POPULATION_SIZE, threads>>> (
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
		matrixMultiplication <<<NSGA2_POPULATION_SIZE, threads>>> (d_temporal_2,
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
		calculateTrace <<<NSGA2_POPULATION_SIZE, 1 >>> (obj, d_temporal_1, d_population_fitness, d_sorted_population_fitness);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "calculateTrace 1 Sync CudaError!\n");
		}

		// Set original index. this is used in sorting operation.
		setOriginalIndex << <NSGA2_POPULATION_SIZE, 1 >> > (d_population_fitness, d_sorted_population_fitness);

		/* Set current population fitness in host memory from device memory */
		cudaMemcpy(h_population_fitness, d_population_fitness,
			NSGA2_POPULATION_SIZE * (OBJECTIVES + 1) * sizeof(unsigned int),
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


/**
 * Set initial values to total dominance, rank and crowding variables (fill with zeros).
 */
__global__ void initializeNSGA2Variables(
	short d_population_total_dominance[], short d_population_rank[], float d_population_crowding[], 
	unsigned int d_population_fitness[][OBJECTIVES + 1], float d_temporal_population_crowding[][3]) {
	d_population_total_dominance[blockIdx.x] = 0;
	d_population_rank[blockIdx.x] = 0;
	d_population_crowding[blockIdx.x] = 0;
	d_temporal_population_crowding[blockIdx.x][0] = 0;
	d_temporal_population_crowding[blockIdx.x][1] = 0;
	d_temporal_population_crowding[blockIdx.x][2] = 0;

	/*
	 * Test values to verify the Pareto fronts calculation, works with POPULATION = 9
	 * must be removed after verification, also d_population_fitness must be removed
	 * in kernel calls and definition
	 *
	d_population_fitness[0][0] = 10;
	d_population_fitness[0][1] = 625;
	d_population_fitness[1][0] = 40;
	d_population_fitness[1][1] = 600;
	d_population_fitness[2][0] = 30;
	d_population_fitness[2][1] = 500;
	d_population_fitness[3][0] = 0;
	d_population_fitness[3][1] = 400;
	d_population_fitness[4][0] = 20;
	d_population_fitness[4][1] = 325;
	d_population_fitness[5][0] = 60;
	d_population_fitness[5][1] = 450;
	d_population_fitness[6][0] = 70;
	d_population_fitness[6][1] = 375;
	d_population_fitness[7][0] = 60;
	d_population_fitness[7][1] = 275;
	d_population_fitness[8][0] = 80;
	d_population_fitness[8][1] = 125;


	d_population_fitness[9][0] = 100;
	d_population_fitness[9][1] = 0;
	d_population_fitness[10][0] = 90;
	d_population_fitness[10][1] = 290;
	d_population_fitness[11][0] = 100;
	d_population_fitness[11][1] = 400;
	d_population_fitness[12][0] = 120;
	d_population_fitness[12][1] = 375;
	d_population_fitness[13][0] = 140;
	d_population_fitness[13][1] = 350;
	d_population_fitness[14][0] = 150;
	d_population_fitness[14][1] = 250;
	d_population_fitness[15][0] = 170;
	d_population_fitness[15][1] = 75;
	d_population_fitness[16][0] = 170;
	d_population_fitness[16][1] = 300;
	d_population_fitness[17][0] = 180;
	d_population_fitness[17][1] = 50;
	/*****************************************/
}


/**
 * This function get the population dominancen matrix for 2-objective problems.
 */
__global__ void get2ObjectivePopulationDominanceMatrix(bool d_population_dominance_matrix[][NSGA2_POPULATION_SIZE], unsigned int d_population_fitness[][OBJECTIVES + 1]) {

	d_population_dominance_matrix[blockIdx.x][threadIdx.x] =
	(
		(d_population_fitness[threadIdx.x][0] <= d_population_fitness[blockIdx.x][0]) &&
		(d_population_fitness[threadIdx.x][1] <= d_population_fitness[blockIdx.x][1])
	) &&
	(
		(d_population_fitness[threadIdx.x][0] < d_population_fitness[blockIdx.x][0]) ||
		(d_population_fitness[threadIdx.x][1] < d_population_fitness[blockIdx.x][1])
	);
}

/**
 * This function get the population dominancen matrix for 3-objective problems.
 */
__global__ void get3ObjectivePopulationDominanceMatrix(bool d_population_dominance_matrix[][NSGA2_POPULATION_SIZE], unsigned int d_population_fitness[][OBJECTIVES + 1]) {

	d_population_dominance_matrix[blockIdx.x][threadIdx.x] =
	(
		(d_population_fitness[threadIdx.x][0] <= d_population_fitness[blockIdx.x][0]) &&
		(d_population_fitness[threadIdx.x][1] <= d_population_fitness[blockIdx.x][1]) &&
		(d_population_fitness[threadIdx.x][2] <= d_population_fitness[blockIdx.x][2])
	) &&
	(
		(d_population_fitness[threadIdx.x][0] < d_population_fitness[blockIdx.x][0]) ||
		(d_population_fitness[threadIdx.x][1] < d_population_fitness[blockIdx.x][1]) ||
		(d_population_fitness[threadIdx.x][2] < d_population_fitness[blockIdx.x][2])
	);

}

__global__ void getParallelTotalDominance(short d_population_total_dominance[], bool d_population_dominance_matrix[][NSGA2_POPULATION_SIZE]) {

	unsigned int sum = 0;
	#pragma unroll
	for (int x = 0; x < NSGA2_POPULATION_SIZE; x++) {
		sum += d_population_dominance_matrix[blockIdx.x][x];
	}
	d_population_total_dominance[blockIdx.x] = sum;
}

__global__ void setRank(int iteration, short d_population_total_dominance[], short d_population_rank[]) {

	if (d_population_total_dominance[blockIdx.x] == 0 && d_population_rank[blockIdx.x] == 0) {
		d_population_rank[blockIdx.x] = iteration;
	}

}

__global__ void cleanDominanceMatrix(int iteration, bool d_population_dominance_matrix[][NSGA2_POPULATION_SIZE], short d_population_rank[]) {

	if (d_population_rank[threadIdx.x] == iteration) {
		d_population_dominance_matrix[blockIdx.x][threadIdx.x] = 0;
	}

}

__global__ void bitonicSortStep(unsigned int d_population_fitness[][OBJECTIVES + 1], int j, int k, short objective) {
	unsigned int i, ixj;
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i ^ j;
	// Threads with the lowest ids sort the array.
	if ((ixj) > i) {
		if ((i & k) == 0) {
			// Sorting ascending.
			if (d_population_fitness[i][objective] > d_population_fitness[ixj][objective]) {
				unsigned int temp0 = d_population_fitness[i][0];
				unsigned int temp1 = d_population_fitness[i][1];
				unsigned int temp2 = d_population_fitness[i][2];

				d_population_fitness[i][0] = d_population_fitness[ixj][0];
				d_population_fitness[i][1] = d_population_fitness[ixj][1];
				d_population_fitness[i][2] = d_population_fitness[ixj][2];

				d_population_fitness[ixj][0] = temp0;
				d_population_fitness[ixj][1] = temp1;
				d_population_fitness[ixj][2] = temp2;

				if (OBJECTIVES == 3) {
					unsigned int temp3= d_population_fitness[i][3];
					d_population_fitness[i][3] = d_population_fitness[ixj][3];
					d_population_fitness[ixj][3] = temp3;
				}
			}
		}
		if ((i & k) != 0) {
			// Sorting descending.
			if (d_population_fitness[i][objective] < d_population_fitness[ixj][objective]) {

				unsigned int temp0 = d_population_fitness[i][0];
				unsigned int temp1 = d_population_fitness[i][1];
				unsigned int temp2 = d_population_fitness[i][2];

				d_population_fitness[i][0] = d_population_fitness[ixj][0];
				d_population_fitness[i][1] = d_population_fitness[ixj][1];
				d_population_fitness[i][2] = d_population_fitness[ixj][2];

				d_population_fitness[ixj][0] = temp0;
				d_population_fitness[ixj][1] = temp1;
				d_population_fitness[ixj][2] = temp2;

				if (OBJECTIVES == 3) {
					unsigned int temp3 = d_population_fitness[i][3];
					d_population_fitness[i][3] = d_population_fitness[ixj][3];
					d_population_fitness[ixj][3] = temp3;
				}
			}
		}
	}
}

/**
 * Desc population sort by the input objective using bitonic sort.
 */
void sortPopulationByFitness(unsigned int d_population_fitness[][OBJECTIVES + 1], unsigned int objective)
{
	// Variable for population fitness in host memory.
	unsigned int h_population_fitness[NSGA2_POPULATION_SIZE][OBJECTIVES + 1];

	/* Set current population fitness in host memory from device memory */
	cudaMemcpy(h_population_fitness, d_population_fitness,
		NSGA2_POPULATION_SIZE * (OBJECTIVES + 1) * sizeof(unsigned int),
		cudaMemcpyDeviceToHost);

	printf("\nUNSORTED fitness ojective %d \n", objective);
	for (int i = 0; i < NSGA2_POPULATION_SIZE; i++) {
		for (int j = 0; j < OBJECTIVES + 1; j++) {
			printf("|%d ", h_population_fitness[i][j]);
		}
		printf("\n");
	}

	int j, k;
	// Major step.
	for (k = 2; k <= NSGA2_POPULATION_SIZE; k <<= 1) {
		// Minor step.
		for (j = k >> 1; j > 0; j = j >> 1) {
			bitonicSortStep <<<POPULATION_SIZE, 2 >> > (d_population_fitness, j, k, objective);
		}
	}

	/* Set current population fitness in host memory from device memory */
	cudaMemcpy(h_population_fitness, d_population_fitness,
		NSGA2_POPULATION_SIZE * (OBJECTIVES + 1) * sizeof(unsigned int),
		cudaMemcpyDeviceToHost);

	printf("\nSORTED fitness ojective %d \n", objective);
	for (int i = 0; i < NSGA2_POPULATION_SIZE; i++) {
		for (int j = 0; j < OBJECTIVES + 1; j++) {
			printf("|%d ", h_population_fitness[i][j]);
		}
		printf("\n");
	}

}

__global__ void crowdingCalcualtion(
	short current_pareto_front, float d_temporal_population_crowding[][3], float d_population_crowding[],
	unsigned int d_population_fitness[][OBJECTIVES + 1], unsigned int d_sorted_population_fitness[][OBJECTIVES + 1],
	short objective) {

	
	if (blockIdx.x == 0 || 
		((blockIdx.x + 1) < NSGA2_POPULATION_SIZE && (blockIdx.x - 1) >= 0 &&
		 d_temporal_population_crowding[blockIdx.x + 1][0] != current_pareto_front &&
		 d_temporal_population_crowding[blockIdx.x][0] == current_pareto_front)
		) {
		 
		d_temporal_population_crowding[blockIdx.x][2] = (unsigned int)HUGE_VALF;
	}
	else if ((blockIdx.x + 1) < NSGA2_POPULATION_SIZE && d_temporal_population_crowding[blockIdx.x + 1][0] == current_pareto_front) {
		d_temporal_population_crowding[blockIdx.x][2] = 
			(float)d_temporal_population_crowding[blockIdx.x][2] + 
			(
				(float)(d_population_fitness[(int)d_temporal_population_crowding[blockIdx.x + 1][1]][objective] - 
				 d_population_fitness[(int)d_temporal_population_crowding[blockIdx.x - 1][1]][objective]) /
				(float)(d_sorted_population_fitness[NSGA2_POPULATION_SIZE - 1][objective] - d_sorted_population_fitness[0][objective])
			);
	}
	d_population_crowding[(int)d_temporal_population_crowding[blockIdx.x][1]] = d_temporal_population_crowding[blockIdx.x][2];

}

/*
 * Set as zero all initial crowding distances.
 */
__global__ void crowdingInitialization(float d_temporal_population_crowding[][3], float d_population_crowding[]) {
	d_population_crowding[blockIdx.x] = 0;
	d_temporal_population_crowding[blockIdx.x][2] = 0;
}


__global__ void paretoFrontBitonicSortStep(float d_temporal_population_crowding[][3], int j, int k) {
	unsigned int i, ixj;
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i ^ j;
	// Threads with the lowest ids sort the array.
	if ((ixj) > i) {
		if ((i & k) == 0) {
			// Sorting ascending.
			if (d_temporal_population_crowding[i][2] > d_temporal_population_crowding[ixj][2]) {
				float temp0 = d_temporal_population_crowding[i][0];
				float temp1 = d_temporal_population_crowding[i][1];
				float temp2 = d_temporal_population_crowding[i][2];

				d_temporal_population_crowding[i][0] = d_temporal_population_crowding[ixj][0];
				d_temporal_population_crowding[i][1] = d_temporal_population_crowding[ixj][1];
				d_temporal_population_crowding[i][2] = d_temporal_population_crowding[ixj][2];

				d_temporal_population_crowding[ixj][0] = temp0;
				d_temporal_population_crowding[ixj][1] = temp1;
				d_temporal_population_crowding[ixj][2] = temp2;

			}
		}
		if ((i & k) != 0) {
			// Sorting descending.
			if (d_temporal_population_crowding[i][2] < d_temporal_population_crowding[ixj][2]) {

				float temp0 = d_temporal_population_crowding[i][0];
				float temp1 = d_temporal_population_crowding[i][1];
				float temp2 = d_temporal_population_crowding[i][2];

				d_temporal_population_crowding[i][0] = d_temporal_population_crowding[ixj][0];
				d_temporal_population_crowding[i][1] = d_temporal_population_crowding[ixj][1];
				d_temporal_population_crowding[i][2] = d_temporal_population_crowding[ixj][2];

				d_temporal_population_crowding[ixj][0] = temp0;
				d_temporal_population_crowding[ixj][1] = temp1;
				d_temporal_population_crowding[ixj][2] = temp2;
			}
		}
	}
}


/**
 * Desc Pareto front sort by the crowding distance using bitonic sort.
 */
void sortParetoFrontByCrowding(float d_temporal_population_crowding[][3])
{
	int j, k;
	// Major step.
	for (k = 2; k <= NSGA2_POPULATION_SIZE; k <<= 1) {
		// Minor step.
		for (j = k >> 1; j > 0; j = j >> 1) {
			paretoFrontBitonicSortStep << <POPULATION_SIZE, 2 >> > (d_temporal_population_crowding, j, k);
		}
	}
}

/*
 * Caculates the crowding distance of all elements in a Parteto Front
 */

short crowding(
	short current_pareto_front,
	unsigned int d_population_fitness[][OBJECTIVES + 1], 
	unsigned int d_sorted_population_fitness[][OBJECTIVES + 1],
	short d_population_rank[],
	short h_population_rank[],
	float d_temporal_population_crowding[][3],
	float d_population_crowding[],
	float h_population_crowding[]) {

	// Variable to store temporarily the Pareto front crowding in host memory.
	// [0] for rank.
	// [1] original index.
	// [2] crowding distance.
	float h_temporal_population_crowding[NSGA2_POPULATION_SIZE][3];

	// Variable for sorted population fitness in host memory.
	unsigned int h_sorted_population_fitness[NSGA2_POPULATION_SIZE][OBJECTIVES + 1] ;

	short top_total;
	short botton_total;

	crowdingInitialization <<<NSGA2_POPULATION_SIZE, 1 >>> (d_temporal_population_crowding, d_population_crowding);

	/* Set current Pareto front crowding in host memory from device memory */
	cudaMemcpy(h_temporal_population_crowding, d_temporal_population_crowding,
		NSGA2_POPULATION_SIZE * 3 * sizeof(float),
		cudaMemcpyDeviceToHost);
	
	cudaMemcpy(h_population_crowding, d_population_crowding,
		NSGA2_POPULATION_SIZE * sizeof(float),
		cudaMemcpyDeviceToHost);
	
	for (short objective = 0; objective < OBJECTIVES; objective++) {
		sortPopulationByFitness(d_sorted_population_fitness, objective);
		/* Set current Sorted by F(i-objective) population in host memory from device memory */
		cudaMemcpy(h_sorted_population_fitness, d_sorted_population_fitness,
			NSGA2_POPULATION_SIZE * (OBJECTIVES + 1) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);

		top_total = botton_total = 0;

		for (short i = 0; i < NSGA2_POPULATION_SIZE; i++) {

			if (h_population_rank[h_sorted_population_fitness[i][OBJECTIVES]] == current_pareto_front) {
				h_temporal_population_crowding[top_total][0] = current_pareto_front;
				h_temporal_population_crowding[top_total][1] = h_sorted_population_fitness[i][OBJECTIVES];
				h_temporal_population_crowding[top_total][2] = h_population_crowding[h_sorted_population_fitness[i][OBJECTIVES]];
				top_total++;
			}
			else {
				h_temporal_population_crowding[NSGA2_POPULATION_SIZE - (botton_total + 1)][0] = h_population_rank[h_sorted_population_fitness[i][OBJECTIVES]];
				h_temporal_population_crowding[NSGA2_POPULATION_SIZE - (botton_total + 1)][1] = h_sorted_population_fitness[i][OBJECTIVES];
				h_temporal_population_crowding[NSGA2_POPULATION_SIZE - (botton_total + 1)][2] = h_population_crowding[h_sorted_population_fitness[i][OBJECTIVES]];
				botton_total++;
			}
		}

		printf("\nTEMPORAL(INICIO) Pareto front %d OBJECTIVE %d \n", current_pareto_front, objective);
		for (int x = 0; x < NSGA2_POPULATION_SIZE; x++) {
			for (int y = 0; y < 3; y++) {
				printf("%f ", h_temporal_population_crowding[x][y]);
			}
			printf("\n");
		}

		// Set population crowding in device memory from host memory.
		cudaMemcpy(d_temporal_population_crowding, h_temporal_population_crowding,
			NSGA2_POPULATION_SIZE * 3 * sizeof(float),
			cudaMemcpyHostToDevice);

		crowdingCalcualtion << <NSGA2_POPULATION_SIZE, 1 >> > (current_pareto_front, d_temporal_population_crowding, d_population_crowding, d_population_fitness, d_sorted_population_fitness, objective);
		
		/* Set population crowding in host memory from device memory */
		cudaMemcpy(h_temporal_population_crowding, d_temporal_population_crowding,
			NSGA2_POPULATION_SIZE * 3 * sizeof(float),
			cudaMemcpyDeviceToHost);
		cudaMemcpy(h_population_crowding, d_population_crowding,
			NSGA2_POPULATION_SIZE * sizeof(float),
			cudaMemcpyDeviceToHost);

		printf("\nTEMPORAL(FIN) Pareto front %d OBJECTIVE %d \n", current_pareto_front, objective);
		for (int x = 0; x < NSGA2_POPULATION_SIZE; x++) {
			for (int y = 0; y < 3; y++) {
				printf("%f ", h_temporal_population_crowding[x][y]);
			}
			printf("\n");
		}

	}

	sortParetoFrontByCrowding(d_temporal_population_crowding);

	return top_total;
}

void parallelNSGA2(
	short h_population[][FACILITIES_LOCATIONS], unsigned int h_population_fitness[][OBJECTIVES + 1],
	short h_population_total_dominance[], short h_population_rank[], float h_population_crowding[],
	short d_population[][FACILITIES_LOCATIONS], unsigned int d_population_fitness[][OBJECTIVES + 1], unsigned int d_sorted_population_fitness[][OBJECTIVES + 1],
	short d_population_total_dominance[], short d_population_rank[], float d_population_crowding[]) {

	/* Variable to check correct synchronization */
	cudaError_t cudaStatus;

	/*******************************************************************************************
	 * Comment this section if you don't need to print partial results of NSGA2.
	 */

	// Variable to store the population dominance in host memory.
	bool h_population_dominance_matrix[NSGA2_POPULATION_SIZE][NSGA2_POPULATION_SIZE];

	/*******************************************************************************************/

	// Variable to store temporarily the Pareto front crowding in host memory.
	// [0] for rank.
	// [1] original index.
	// [2] crowding distance.
	float h_temporal_population_crowding[NSGA2_POPULATION_SIZE][3];

	// Variable to store the the total count of selected solution in each iteration.
	short offspring_count = 0;

	// Number of solutions in the Parto front.
	short pareto_front_count = 0;

	// Variable to store the population dominancen in device memory.
	bool(*d_population_dominance_matrix)[NSGA2_POPULATION_SIZE];
	cudaMalloc((void**)&d_population_dominance_matrix,
		sizeof(bool) * NSGA2_POPULATION_SIZE * NSGA2_POPULATION_SIZE);

	// Variable to store temporarily the population crowding in device memory.
	// will be use for crauding calculation and offspring selection
	// [0] for rank.
	// [1] original index.
	// [2] crowding distance.
	float(*d_temporal_population_crowding)[3];
	cudaMalloc((void**)&d_temporal_population_crowding,
		sizeof(float) * NSGA2_POPULATION_SIZE * 3);

	/*
	 * Set initial values to totaldominance, rank and crowding variables (fill with zeros).
	 */
	initializeNSGA2Variables <<<NSGA2_POPULATION_SIZE, 1 >>> (d_population_total_dominance, d_population_rank, d_population_crowding, d_population_fitness, d_temporal_population_crowding);

	/*
	 * calculate the populaton dominace matrix.
	 */
	if (OBJECTIVES == 2) {
		get2ObjectivePopulationDominanceMatrix <<<NSGA2_POPULATION_SIZE, NSGA2_POPULATION_SIZE>>> (d_population_dominance_matrix, d_population_fitness);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "get2ObjectivePopulationDominanceMatrix Sync CudaError!\n");
		}
	}
	else if (OBJECTIVES == 3) {
		get3ObjectivePopulationDominanceMatrix <<<NSGA2_POPULATION_SIZE, NSGA2_POPULATION_SIZE>>> (d_population_dominance_matrix, d_population_fitness);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "get3ObjectivePopulationDominanceMatrix Sync CudaError!\n");
		}
	}
	else {
		printf("\nThis solution only suports 2 and 3 objetives QAP\n");
		exit(0);
	}

	/*
	 * calculate the total dominance per solution.
	 */
	getParallelTotalDominance <<<NSGA2_POPULATION_SIZE, 1 >>> (d_population_total_dominance, d_population_dominance_matrix);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "getParallelTotalDominance Sync CudaError!\n");
	}


	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of  population dominace matrix calculation.
	 */

	// Set current population dominance matrix in host memory from device memory.
	cudaMemcpy(h_population_dominance_matrix, d_population_dominance_matrix,
		NSGA2_POPULATION_SIZE * NSGA2_POPULATION_SIZE * sizeof(bool),
		cudaMemcpyDeviceToHost);
	// Set current population total dominance in host memory from device memory.
	cudaMemcpy(h_population_total_dominance, d_population_total_dominance,
		NSGA2_POPULATION_SIZE * sizeof(short),
		cudaMemcpyDeviceToHost);

	printf("\nPopulation dominace matrix\n");
	for (int i = 0; i < NSGA2_POPULATION_SIZE; i++) {
		for (int j = 0; j < NSGA2_POPULATION_SIZE; j++) {
			printf("%d ", h_population_dominance_matrix[i][j]);
		}
		printf("| %d\n", h_population_total_dominance[i]);
		printf("\n");
	}
	/*********************************************************************/


	// Main Routine to calculate Pareto Fronts (until NSGA2_POPULATION_SIZE, the worst case or offspring population completed).
	for (int i = 1; i < NSGA2_POPULATION_SIZE; i++) {
		// Set NSGA2 Rank.
		setRank <<<NSGA2_POPULATION_SIZE,1 >>> (i, d_population_total_dominance, d_population_rank);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "setRank Sync CudaError!\n");
		}

		// Set current population ranck in host memory from device memory. (MANDATORY) 
		cudaMemcpy(h_population_rank, d_population_rank,
			NSGA2_POPULATION_SIZE * sizeof(short),
			cudaMemcpyDeviceToHost);

		/*********************************************************************/
		printf("\nPopulation Rank \n");
		for (int x = 0; x < NSGA2_POPULATION_SIZE; x++) {
			printf("%d \n", h_population_rank[x]);
		}
		/*********************************************************************/

		// Calculate Pareto front Crowding distance
		pareto_front_count = crowding(i, d_population_fitness, d_sorted_population_fitness, d_population_rank, h_population_rank, d_temporal_population_crowding, d_population_crowding, h_population_crowding);

		/*********************************************************************
		 * Comment this section if you don't need to print the partial results
		 * of Pareto front sorted by crowding distance.
		 */

		 // Set current Pareto front sorted by crowding distance in host memory from device memory.
		cudaMemcpy(h_temporal_population_crowding, d_temporal_population_crowding,
			NSGA2_POPULATION_SIZE * 3 * sizeof(float),
			cudaMemcpyDeviceToHost);

		printf("\nPareto front %d sorted by crowding distance\n", i);
		for (int x = 0; x < NSGA2_POPULATION_SIZE; x++) {
			for (int y = 0; y < 3; y++) {
				printf("%f ", h_temporal_population_crowding[x][y]);
			}
			printf("\n");
		}
		/*********************************************************************/

		// @todo Move selected pareto front solutions to the offspring.
		//moveSolutionsToOffspring <<<NSGA2_POPULATION_SIZE, pareto_front_count >> > (offspring_count, pareto_front_count, d_population, d_temporal_population_crowding);
		//moveRankToOffspring <<<NSGA2_POPULATION_SIZE, 1 >> > (offspring_count, pareto_front_count, d_rank, d_temporal_population_crowding);
		//moveCrowdingToOffspring <<<NSGA2_POPULATION_SIZE, 1 >> > (offspring_count, pareto_front_count, d_population_crowding, d_temporal_population_crowding);

		offspring_count += pareto_front_count;

		printf("\nPareto front %d count: %d\n", i, pareto_front_count);
		printf("\nPareto front %d OFFSpring count: %d\n", i, offspring_count);

		// End the loop if the offspring is complete.
		if (offspring_count >= POPULATION_SIZE) {
			break;
		}

		// Remove the current Pareto Front elements from the dominance matrix.
		cleanDominanceMatrix <<<NSGA2_POPULATION_SIZE, NSGA2_POPULATION_SIZE >>> (i, d_population_dominance_matrix, d_population_rank);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cleanDominanceMatrix Sync CudaError!\n");
		}

		// Calculate the new total dominance per solution.
		getParallelTotalDominance <<<NSGA2_POPULATION_SIZE, 1 >>> (d_population_total_dominance, d_population_dominance_matrix);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "getParallelTotalDominance in NSGA2 iteration Sync CudaError!\n");
		}

	}
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
	unsigned int h_population_fitness[NSGA2_POPULATION_SIZE][OBJECTIVES + 1];
	// Variable for population total dominace in host memory.
	short h_population_total_dominance[NSGA2_POPULATION_SIZE];
	// Variable for population rank in host memory.
	short h_population_rank[NSGA2_POPULATION_SIZE];
	// Variable for population crowding distance in host memory.
	float h_population_crowding[NSGA2_POPULATION_SIZE];

	/**********VARIABLES IN DEVICE MEMORY********/
	// Variable for population in device memory.
	short(*d_population)[FACILITIES_LOCATIONS];
	cudaMalloc((void**)&d_population, sizeof(short) * NSGA2_POPULATION_SIZE * FACILITIES_LOCATIONS);
	// Variable for population fitness in device memory.
	unsigned int(*d_population_fitness)[OBJECTIVES + 1];
	cudaMalloc((void**)&d_population_fitness, sizeof(unsigned int) * NSGA2_POPULATION_SIZE * (OBJECTIVES + 1));
	// Variable for sorted population fitness in device memory.
	unsigned int(*d_sorted_population_fitness)[OBJECTIVES + 1];
	cudaMalloc((void**)&d_sorted_population_fitness, sizeof(unsigned int) * NSGA2_POPULATION_SIZE * (OBJECTIVES + 1));
	// Variable for population total dominace in device memory.
	short(*d_population_total_dominance);
	cudaMalloc((void**)&d_population_total_dominance, sizeof(short) * NSGA2_POPULATION_SIZE);
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
	 *
	
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

	*/

	h_population[0][0] =1; h_population[0][1] =9; h_population[0][2] =6; h_population[0][3] =2; h_population[0][4] =8; h_population[0][5] =0; h_population[0][6] =4; h_population[0][7] =3; h_population[0][8] =5; h_population[0][9] =7;
	h_population[1][0] =3; h_population[1][1] =7; h_population[1][2] =4; h_population[1][3] =6; h_population[1][4] =2; h_population[1][5] =9; h_population[1][6] =1; h_population[1][7] =8; h_population[1][8] =0; h_population[1][9] =5;
	h_population[2][0] =4; h_population[2][1] =9; h_population[2][2] =8; h_population[2][3] =5; h_population[2][4] =6; h_population[2][5] =0; h_population[2][6] =7; h_population[2][7] =3; h_population[2][8] =2; h_population[2][9] =1;
	h_population[3][0] =2; h_population[3][1] =9; h_population[3][2] =1; h_population[3][3] =5; h_population[3][4] =4; h_population[3][5] =0; h_population[3][6] =3; h_population[3][7] =8; h_population[3][8] =6; h_population[3][9] =7;
	h_population[4][0] =6; h_population[4][1] =9; h_population[4][2] =0; h_population[4][3] =1; h_population[4][4] =8; h_population[4][5] =4; h_population[4][6] =3; h_population[4][7] =2; h_population[4][8] =5; h_population[4][9] =7;
	h_population[5][0] =8; h_population[5][1] =7; h_population[5][2] =5; h_population[5][3] =1; h_population[5][4] =4; h_population[5][5] =6; h_population[5][6] =9; h_population[5][7] =2; h_population[5][8] =0; h_population[5][9] =3;
	h_population[6][0] =4; h_population[6][1] =9; h_population[6][2] =3; h_population[6][3] =6; h_population[6][4] =0; h_population[6][5] =7; h_population[6][6] =8; h_population[6][7] =5; h_population[6][8] =2; h_population[6][9] =1;
	h_population[7][0] =3; h_population[7][1] =5; h_population[7][2] =1; h_population[7][3] =6; h_population[7][4] =7; h_population[7][5] =0; h_population[7][6] =9; h_population[7][7] =2; h_population[7][8] =8; h_population[7][9] =4;
	h_population[8][0] =6; h_population[8][1] =9; h_population[8][2] =1; h_population[8][3] =8; h_population[8][4] =2; h_population[8][5] =5; h_population[8][6] =7; h_population[8][7] =0; h_population[8][8] =4; h_population[8][9] =3;
	h_population[9][0] =4; h_population[9][1] =8; h_population[9][2] =7; h_population[9][3] =6; h_population[9][4] =9; h_population[9][5] =5; h_population[9][6] =1; h_population[9][7] =0; h_population[9][8] =3; h_population[9][9] =2;
	h_population[10][0] =8; h_population[10][1] =5; h_population[10][2] =4; h_population[10][3] =2; h_population[10][4] =0; h_population[10][5] =1; h_population[10][6] =7; h_population[10][7] =9; h_population[10][8] =3; h_population[10][9] =6;
	h_population[11][0] =3; h_population[11][1] =8; h_population[11][2] =0; h_population[11][3] =6; h_population[11][4] =2; h_population[11][5] =7; h_population[11][6] =4; h_population[11][7] =9; h_population[11][8] =5; h_population[11][9] =1;
	h_population[12][0] =1; h_population[12][1] =6; h_population[12][2] =5; h_population[12][3] =9; h_population[12][4] =8; h_population[12][5] =3; h_population[12][6] =0; h_population[12][7] =2; h_population[12][8] =4; h_population[12][9] =7;
	h_population[13][0] =2; h_population[13][1] =6; h_population[13][2] =3; h_population[13][3] =4; h_population[13][4] =9; h_population[13][5] =0; h_population[13][6] =1; h_population[13][7] =8; h_population[13][8] =5; h_population[13][9] =7;
	h_population[14][0] =4; h_population[14][1] =9; h_population[14][2] =5; h_population[14][3] =7; h_population[14][4] =2; h_population[14][5] =6; h_population[14][6] =0; h_population[14][7] =8; h_population[14][8] =1; h_population[14][9] =3;
	h_population[15][0] =6; h_population[15][1] =0; h_population[15][2] =9; h_population[15][3] =8; h_population[15][4] =2; h_population[15][5] =4; h_population[15][6] =7; h_population[15][7] =3; h_population[15][8] =1; h_population[15][9] =5;

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
		parallelPopulationFitnessCalculation(h_population, h_population_fitness, d_population, d_population_fitness, d_sorted_population_fitness);
		parallelNSGA2(h_population, h_population_fitness, h_population_total_dominance, h_population_rank, h_population_crowding, d_population, d_population_fitness, d_sorted_population_fitness, d_population_total_dominance, d_population_rank, d_population_crowding);

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
		NSGA2_POPULATION_SIZE * (OBJECTIVES + 1) * sizeof(unsigned int),
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

