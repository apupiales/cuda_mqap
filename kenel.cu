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
		int population[][FACILITIES_LOCATIONS + OBJECTIVES + 1]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
//asignar la memoria a 32
	//if (i < POPULATION_SIZE && j < FACILITIES_LOCATIONS + OBJECTIVES + 1) {
		population[i][j] = 0;
		if (j < FACILITIES_LOCATIONS) {
			population[i][j] = j;
		}
	//}
}

/**
 * This function shuffles chromosomes genes randomly over all population
 */
__global__ void shufflePopulationGenes(curandState *my_curandstate,
		const unsigned *max_rand_int, const unsigned *min_rand_int,
		int population[][FACILITIES_LOCATIONS + OBJECTIVES + 1]) {

	int i = blockIdx.x;
	if (i < POPULATION_SIZE) {
#pragma unroll
 for (int j = 0; j < FACILITIES_LOCATIONS; j++) {
		int idx = j + blockDim.x * blockIdx.x;

		float myrandf = curand_uniform(my_curandstate + idx);
		int myrand = int(myrandf * 10);
		//revisar instucción swap de cuda
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
		int population[][FACILITIES_LOCATIONS + OBJECTIVES + 1],
		int population_2d[][FACILITIES_LOCATIONS],
		int population_2d_transposed[][FACILITIES_LOCATIONS]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;

	if (i < POPULATION_SIZE && j < FACILITIES_LOCATIONS
			&& k < FACILITIES_LOCATIONS) {
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
		int population[][FACILITIES_LOCATIONS + OBJECTIVES + 1]) {

	int i = blockIdx.x;

	if (i < POPULATION_SIZE) {
		int sum = 0;
		for (int x = 0; x < FACILITIES_LOCATIONS; x++) {
			sum += input_matrix[x + (i * FACILITIES_LOCATIONS)][x];
		}
		population[i][FACILITIES_LOCATIONS + objective_id] = sum;
	}
}

__global__ void atomicCalculateTrace(int objective_id,
		int input_matrix[][FACILITIES_LOCATIONS],
		int population[][FACILITIES_LOCATIONS + OBJECTIVES + 1]) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;

	if (i < POPULATION_SIZE && j < FACILITIES_LOCATIONS
			&& k < FACILITIES_LOCATIONS) {
		if (j == k) {
			atomicAdd(&population[i][FACILITIES_LOCATIONS + objective_id], input_matrix[j + (i * FACILITIES_LOCATIONS)][k]);
		}
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
		int h_population[][FACILITIES_LOCATIONS + OBJECTIVES + 1], int d_population[][FACILITIES_LOCATIONS + OBJECTIVES + 1]) {

	/* Variable to check correct synchronization */
	cudaError_t cudaStatus;

	/*******************************************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 *

	// Variable for population binary 2d representation in host memory (X)
	int h_2d_population[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable for population binary 2d representation transposed in host memory (XT)
	int h_2d_transposed_population[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep F1*X result in host memory (F1: Flow matrix 1)
	int h_temporal_1[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep F2*X result in host memory (F2: Flow matrix 2)
	int h_temporal_2[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep F1*X*DT result in host memory (DT: Transposed Distances matrix)
	int h_temporal_3[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep F2*X*DT result in host memory (DT: Transposed Distances matrix)
	int h_temporal_4[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep F1*X*DT*XT result in host memory
	int h_temporal_5[POPULATION_SIZE * FACILITIES_LOCATIONS][FACILITIES_LOCATIONS];

	// Variable to keep F2*X*DT*XT result in host memory
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
	 * Variable to keep F2*X result in device memory (F2: Flow matrix 2).
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
		fprintf(stderr, "populationTo2DRepresentation Sync CudaError!\n");
	}

	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 *

	// Set current population binary 2d representation in host memory from device memory
	cudaMemcpy(h_2d_population, d_2d_population,
	POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(int),
			cudaMemcpyDeviceToHost);
	// Set current population binary 2d representation transposed in host memory from device memory
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
		fprintf(stderr, "multiplicationWithFlowMatrix 1 Sync CudaError!\n");
	}
	/*
	 * F2*X
	 */
	multiplicationWithFlowMatrix<<<POPULATION_SIZE, threads>>>(1,
			d_2d_population, d_temporal_2);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multiplicationWithFlowMatrix 2 Sync CudaError!\n");
	}

	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 *

	// Set the result of F1*X in host memory from device memory
	cudaMemcpy(h_temporal_1, d_temporal_1,
			POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS
					* sizeof(int), cudaMemcpyDeviceToHost);
	// Set the result of F2*X in host memory from device memory
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
		fprintf(stderr, "multiplicationWithTranposedDistanceMatrix 1 Sync CudaError!\n");
	}
	/*
	 * F2*X*DT
	 */
	multiplicationWithTranposedDistanceMatrix<<<POPULATION_SIZE, threads>>>(
			d_temporal_2, d_temporal_4);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multiplicationWithTranposedDistanceMatrix 2 Sync CudaError!\n");
	}

	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 *

	// Set the result of F1*X*DT in host memory from device memory
	cudaMemcpy(h_temporal_3, d_temporal_3,
	POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(int),
			cudaMemcpyDeviceToHost);
	// Set the result of F2*X*DT in host memory from device memory
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
		fprintf(stderr, "matrixMultiplication 1 Sync CudaError!\n");
	}
	/*
	 * F2*X*DT*XT
	 */
	matrixMultiplication<<<POPULATION_SIZE, threads>>>(d_temporal_4,
			d_2d_transposed_population, d_temporal_2);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "matrixMultiplication 2 Sync CudaError!\n");
	}

	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 *

	// Set the result of F1*X*DT*XT in host memory from device memory
	cudaMemcpy(h_temporal_5, d_temporal_1,
	POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(int),
			cudaMemcpyDeviceToHost);
	// Set the result of F2*X*DT*XT in host memory from device memory
	cudaMemcpy(h_temporal_6, d_temporal_1,
	POPULATION_SIZE * FACILITIES_LOCATIONS * FACILITIES_LOCATIONS * sizeof(int),
			cudaMemcpyDeviceToHost);
	/*********************************************************************/

	/*
	 * Trace(F1*X*DT*XT)
	 */
	calculateTrace<<<POPULATION_SIZE, 1>>>(0, d_temporal_1, d_population);
	//atomicCalculateTrace<<<POPULATION_SIZE, threads>>>(0, d_temporal_1, d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calculateTrace 1 Sync CudaError!\n");
	}
	/*
	 * Trace(F2*X*DT*XT)
	 */
	calculateTrace<<<POPULATION_SIZE, 1>>>(1, d_temporal_2, d_population);
	//atomicCalculateTrace<<<POPULATION_SIZE, threads>>>(1, d_temporal_2, d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calculateTrace 2 Sync CudaError!\n");
	}

	/* Set current population with fitness in host memory from device memory */
	cudaMemcpy(h_population, d_population,
	POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES + 1) * sizeof(int),
			cudaMemcpyDeviceToHost);


	/*********************************************************************
	 * Comment this section if you don't need to print the partial results
	 * of fitness calculation.
	 * NOTE: If you uncomment this section, all previous sections with this
	 * notes must be uncomented too.
	 *
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

__global__ void calculateTotalFitness(
		int population[][FACILITIES_LOCATIONS + OBJECTIVES + 1],
		int *d_population_fitness_sum) {

	int i = blockIdx.x;

	if (i < POPULATION_SIZE) {
		int sum = 0;
		for (int x = 0; x < OBJECTIVES; x++) {
			sum += population[i][FACILITIES_LOCATIONS + x];
		}
		population[i][FACILITIES_LOCATIONS + OBJECTIVES] = sum;
		atomicAdd(d_population_fitness_sum, sum);
	}
}

__global__ void calculatePopulationProbability(
		int population[][FACILITIES_LOCATIONS + OBJECTIVES + 1],
		int population_fitness_sum, float d_population_probability[],
		int d_population_times_selected[]) {

	int i = blockIdx.x;

	if (i < POPULATION_SIZE) {
		d_population_times_selected[i] = 0;
		d_population_probability[i] = (1.0
				/ ((float)population[i][FACILITIES_LOCATIONS + OBJECTIVES]
						/ (float)population_fitness_sum)) / (float)POPULATION_SIZE;

		printf("%d: %f\n", i,d_population_probability[i]);

		while (d_population_probability[i] >= 1) {
			d_population_times_selected[i]++;
			d_population_probability[i]--;

		}

	}
}

__global__ void elitistSelection(
		int population[][FACILITIES_LOCATIONS + OBJECTIVES + 1],
		int population_selected[][FACILITIES_LOCATIONS + OBJECTIVES + 1],
		int d_population_times_selected[],
		int d_number_of_selections) {

	int * temp_chromosome;
	cudaMalloc((void**) &temp_chromosome,
			sizeof(int) * (FACILITIES_LOCATIONS + OBJECTIVES + 1));

	d_number_of_selections = 0;
	for (int i = 0; i < POPULATION_SIZE; i++) {
		for (int j = 0; j < d_population_times_selected[i]; j++) {
			memcpy(population_selected[d_number_of_selections], population[i], sizeof(int) * (FACILITIES_LOCATIONS + OBJECTIVES + 1));
			d_number_of_selections++;
		}
	}
	//release tem_chromosome

}


__global__ void sortCandidatesByRemainingProbability(
		int population[][FACILITIES_LOCATIONS + OBJECTIVES + 1],
		float d_population_probability[]) {

	int * temp_chromosome;
		cudaMalloc((void**) &temp_chromosome,
				sizeof(int) * (FACILITIES_LOCATIONS + OBJECTIVES + 1));

	for (int i = 0; i < POPULATION_SIZE; i++) {
		for (int j = i + 1; j < POPULATION_SIZE; j++) {
			if (d_population_probability[i] < d_population_probability[j]) {
				float temp = d_population_probability[i];
				d_population_probability[i] = d_population_probability[j];
				d_population_probability[j] = temp;
				memcpy(temp_chromosome, population[i], sizeof(int) * (FACILITIES_LOCATIONS + OBJECTIVES + 1));
				memcpy(population[i], population[j], sizeof(int) * (FACILITIES_LOCATIONS + OBJECTIVES + 1));
				memcpy(population[j], temp_chromosome, sizeof(int) * (FACILITIES_LOCATIONS + OBJECTIVES + 1));
			}
		}
	}
	//release tem_chromosome
}


__global__ void rouletteSelection(
		int population[][FACILITIES_LOCATIONS + OBJECTIVES + 1],
		int population_selected[][FACILITIES_LOCATIONS + OBJECTIVES + 1],
		float d_population_probability[],
		curandState *selection_curandstate,
		int x) {

	for (int i = 0; x < POPULATION_SIZE; i++) {
		int idx = i % (POPULATION_SIZE * 32);
		i %= POPULATION_SIZE;
		float random = curand_uniform(&selection_curandstate[idx]);
		if (0.4 < 0.5) {
			memcpy(population_selected[x], population[i], sizeof(int) * (FACILITIES_LOCATIONS + OBJECTIVES + 1));
			x++;
			if (x == POPULATION_SIZE) {
				break;
			}
		}
	}

	memcpy(population, population_selected, sizeof(int) * POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES + 1));

}




void selection(
		int h_population[][FACILITIES_LOCATIONS + OBJECTIVES + 1],
		int d_population[][FACILITIES_LOCATIONS + OBJECTIVES + 1],
		curandState *selection_curandstate) {

	/* Variable to check correct synchronization */
	cudaError_t cudaStatus;

	/* Variable for population fitness sum in device memory */
	int h_population_fitness_sum = 0;

	int h_number_of_selections = 0;

	/* Variable for population fitness sum in device memory  */
	int * d_population_fitness_sum;
	cudaMalloc((void**)&d_population_fitness_sum, sizeof(int));
	cudaMemcpy(d_population_fitness_sum, &h_population_fitness_sum, sizeof(int), cudaMemcpyHostToDevice);

	/* Variable to keep the selection population probability in device memory  */
	float * d_population_probability;
	cudaMalloc((void**) &d_population_probability,
			POPULATION_SIZE * sizeof(float));

	/* Variable to keep the selection population probability in device memory  */
	int * d_population_times_selected;
	cudaMalloc((void**) &d_population_times_selected,
			POPULATION_SIZE * sizeof(int));

	/* Variable for population in device memory */
	int (*d_population_selected)[FACILITIES_LOCATIONS + OBJECTIVES + 1];
	cudaMalloc((void**) &d_population_selected,
			sizeof(int) * POPULATION_SIZE
					* (FACILITIES_LOCATIONS + OBJECTIVES + 1));

	/*
	 * Fitness 1 plus fitness 2
	 */
	calculateTotalFitness<<<POPULATION_SIZE, 1>>>(d_population, d_population_fitness_sum);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calculateTotalFitness Sync CudaError!\n");
	}

	cudaMemcpy(&h_population_fitness_sum, d_population_fitness_sum, sizeof(int), cudaMemcpyDeviceToHost);

	printf("Fitness sum: %d\n",h_population_fitness_sum);

	calculatePopulationProbability<<<POPULATION_SIZE, 1>>>(d_population, h_population_fitness_sum, d_population_probability, d_population_times_selected);
	cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "calculatePopulationProbability Sync CudaError!\n");
		}

	elitistSelection<<<1, 1>>>(d_population, d_population_selected, d_population_times_selected, h_number_of_selections);
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "elitistSelection Sync CudaError!\n");
			}
	sortCandidatesByRemainingProbability<<<1, 1>>>(d_population,
			d_population_probability);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
				"sortCandidatesByRemainingProbability Sync CudaError!\n");
	}

	rouletteSelection<<<1, 1>>>(d_population, d_population_selected, d_population_probability, selection_curandstate, h_number_of_selections);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "rouletteSelection Sync CudaError!\n");
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
	int h_population[POPULATION_SIZE][FACILITIES_LOCATIONS + OBJECTIVES + 1];

	/* Variable for population in device memory */
	int (*d_population)[FACILITIES_LOCATIONS + OBJECTIVES + 1];
	cudaMalloc((void**) &d_population,
			sizeof(int) * POPULATION_SIZE
					* (FACILITIES_LOCATIONS + OBJECTIVES + 1));

	/* Generation of all base chromosomes (genes ordered ascending) */
	generateBasePopulation<<<POPULATION_SIZE, 32>>>(d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "generateBasePopulation Sync CudaError!\n");
	}

	/* Uncomment this section of code to print the base population
	 Set population in host memory from device memory
	cudaMemcpy(h_population, d_population,
	POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES + 1) * sizeof(int),
			cudaMemcpyDeviceToHost);
	 printf("\nBase Population\n");
	 for (int i = 0; i < POPULATION_SIZE; i++) {
	 printf("Chromosome %d\n", i);
	 for (int j = 0; j < FACILITIES_LOCATIONS + OBJECTIVES + 1; j++) {
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

	curand_setup<<<POPULATION_SIZE, 16>>>(d_state, seed);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "curand_setup Sync CudaError!");
	}

	/* Shuffles chromosome genes randomly over all population */
	shufflePopulationGenes<<<POPULATION_SIZE, 1>>>(d_state, d_max_rand_int,
			d_min_rand_int, d_population);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "shufflePopulationGenes Sync CudaError!");
	}
	/* Uncomment this section of code to print the Shuffled population
	 Set current population (with Shuffled genes) in host memory from device memory
	cudaMemcpy(h_population, d_population,
	POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES + 1) * sizeof(int),
			cudaMemcpyDeviceToHost);

	 printf("\nShuffled Population\n");
	 for (int i = 0; i < POPULATION_SIZE; i++) {
	 printf("Chromosome %d\n", i);
	 for (int j = 0; j < FACILITIES_LOCATIONS + OBJECTIVES + 1; j++) {
	 printf("%d ", h_population[i][j]);
	 }
	 printf("\n");
	 }
	 */

	/* Set all chromosomes with 1 2 7 9 6 5 0 4 3 8 0 0 for test purposes
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
		h_population[a][12] = 0;
	}

	cudaMemcpy(d_population, h_population,
			POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES + 1) * sizeof(int),
				cudaMemcpyHostToDevice);
				*/

	/* Uncomment this section of code to print the Final population */

	/* Set Initial population in host memory from device memory */
	cudaMemcpy(h_population, d_population,
	POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES + 1) * sizeof(int),
			cudaMemcpyDeviceToHost);

	printf("\nInitial Population\n");
	for (int i = 0; i < POPULATION_SIZE; i++) {
		printf("Chromosome %d\n", i);
		for (int j = 0; j < FACILITIES_LOCATIONS + OBJECTIVES + 1; j++) {
			printf("%d ", h_population[i][j]);
		}
		printf("\n");
	}
	/******************************************************************/

	for (int iteration = 1; iteration <= ITERATIONS; iteration++) {

		/* Calculate fitness on each population chromosome */
		calculatePopulationfitness(h_population, d_population);
		/* Selection operation */
		//selection(h_population, d_population, d_state);

	}

	/* Set current population in host memory from device memory */
	cudaMemcpy(h_population, d_population,
	POPULATION_SIZE * (FACILITIES_LOCATIONS + OBJECTIVES + 1) * sizeof(int),
			cudaMemcpyDeviceToHost);

	/* Uncomment this section of code to print the Final population */
	printf("\nFinal Population\n");
	for (int i = 0; i < POPULATION_SIZE; i++) {
		printf("Chromosome %d\n", i);
		for (int j = 0; j < FACILITIES_LOCATIONS + OBJECTIVES + 1; j++) {
			printf("%d ", h_population[i][j]);
		}
		printf("\n");
	}
	/******************************************************************/

	clock_t end = clock();
	double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
	printf("\n Time Spent: %f", time_spent);

	exit(0);
}

