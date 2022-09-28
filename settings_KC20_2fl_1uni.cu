/*
 * Executed in a "GeForce RTX 2060M" with
 * (30) Multiprocessors, (64) CUDA Cores/MP.
 */

/*
 * Number of Facilities/locations in the mQAP problem.
 */
#define FACILITIES_LOCATIONS 20

/*
 * Number of objectives in the mQAP problem.
 */
#define OBJECTIVES 2

/*
 * Pt population size. MUST BE A POWER OF TWO
 * due to bitonic sort limitation.
 */
#define POPULATION_SIZE 64

/*
 * Number of times the genetic algorithm is executed.
 */
#define ITERATIONS 300

/*
* Probability to execute an exchange mutation in a single population member.
*/
#define EXCHANGE_MUTATION_PROBABILITY 1

/*
* Probability to execute transposition mutation in a single population member.
*/
#define TRANSPOSITION_MUTATION_PROBABILITY 1


/*
 * Variable with transpose distances matrix in constant device memory.
 * This transpose distances matrix correspond with KC10-2fl-1uni.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_transposeDistancesMatrix[FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
		{0, 37, 37, 43, 22, 9, 90, 41, 50, 21, 93, 62, 19, 82, 94, 58, 85, 53, 57, 45},
		{37, 0, 98, 15, 77, 27, 15, 20, 1, 38, 27, 58, 4, 57, 39, 89, 62, 88, 26, 61},
		{37, 98, 0, 21, 29, 62, 46, 86, 20, 35, 41, 64, 83, 72, 77, 33, 27, 40, 66, 54},
		{43, 15, 21, 0, 10, 37, 2, 40, 24, 94, 14, 36, 99, 10, 62, 86, 35, 23, 5, 99},
		{22, 77, 29, 10, 0, 98, 15, 78, 89, 6, 63, 100, 71, 13, 79, 64, 89, 22, 32, 93},
		{9, 27, 62, 37, 98, 0, 24, 55, 71, 21, 6, 3, 56, 88, 72, 53, 78, 28, 99, 70},
		{90, 15, 46, 2, 15, 24, 0, 85, 33, 12, 87, 37, 76, 71, 78, 56, 23, 64, 73, 100},
		{41, 20, 86, 40, 78, 55, 85, 0, 95, 84, 75, 21, 60, 71, 98, 77, 19, 56, 63, 6},
		{50, 1, 20, 24, 89, 71, 33, 95, 0, 100, 73, 24, 62, 74, 91, 24, 18, 44, 97, 22},
		{21, 38, 35, 94, 6, 21, 12, 84, 100, 0, 88, 18, 67, 67, 22, 2, 22, 76, 15, 45},
		{93, 27, 41, 14, 63, 6, 87, 75, 73, 88, 0, 70, 15, 93, 36, 51, 38, 17, 45, 93},
		{62, 58, 64, 36, 100, 3, 37, 21, 24, 18, 70, 0, 62, 75, 11, 95, 72, 62, 54, 100},
		{19, 4, 83, 99, 71, 56, 76, 60, 62, 67, 15, 62, 0, 65, 91, 66, 87, 30, 10, 67},
		{82, 57, 72, 10, 13, 88, 71, 71, 74, 67, 93, 75, 65, 0, 46, 62, 35, 48, 95, 21},
		{94, 39, 77, 62, 79, 72, 78, 98, 91, 22, 36, 11, 91, 46, 0, 27, 6, 64, 84, 76},
		{58, 89, 33, 86, 64, 53, 56, 77, 24, 2, 51, 95, 66, 62, 27, 0, 17, 82, 70, 49},
		{85, 62, 27, 35, 89, 78, 23, 19, 18, 22, 38, 72, 87, 35, 6, 17, 0, 34, 15, 52},
		{53, 88, 40, 23, 22, 28, 64, 56, 44, 76, 17, 62, 30, 48, 64, 82, 34, 0, 51, 55},
		{57, 26, 66, 5, 32, 99, 73, 63, 97, 15, 45, 54, 10, 95, 84, 70, 15, 51, 0, 58},
		{45, 61, 54, 99, 93, 70, 100, 6, 22, 45, 93, 100, 67, 21, 76, 49, 52, 55, 58, 0}
};

/*
 * Variable with flow matrices in constant device memory.
 * These flow matrices correspond with KC10-2fl-1uni.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://raw.githubusercontent.com/fredizzimo/keyboardlayout/master/tests/mQAPData/KC10-2fl-1uni.dat
 */
__constant__ int d_flowMatrices[OBJECTIVES][FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{
		{0, 58, 84, 8, 46, 74, 92, 100, 85, 44, 35, 36, 51, 1, 19, 90, 14, 96, 92, 25},
		{58, 0, 66, 38, 43, 50, 33, 33, 37, 93, 19, 30, 65, 74, 94, 60, 55, 68, 26, 49},
		{84, 66, 0, 97, 31, 42, 72, 51, 70, 56, 35, 82, 45, 24, 12, 34, 86, 91, 13, 76},
		{8, 38, 97, 0, 40, 81, 71, 31, 71, 6, 45, 82, 57, 45, 66, 91, 14, 26, 52, 68},
		{46, 43, 31, 40, 0, 94, 89, 58, 66, 17, 80, 8, 71, 39, 49, 45, 19, 45, 35, 1},
		{74, 50, 42, 81, 94, 0, 88, 72, 93, 84, 73, 8, 25, 71, 91, 53, 99, 60, 18, 83},
		{92, 33, 72, 71, 89, 88, 0, 90, 91, 5, 46, 49, 5, 78, 69, 12, 18, 55, 46, 63},
		{100, 33, 51, 31, 58, 72, 90, 0, 85, 89, 48, 70, 70, 65, 16, 84, 50, 68, 23, 54},
		{85, 37, 70, 71, 66, 93, 91, 85, 0, 63, 81, 89, 59, 34, 25, 88, 62, 86, 77, 68},
		{44, 93, 56, 6, 17, 84, 5, 89, 63, 0, 34, 5, 87, 8, 91, 15, 46, 83, 75, 27},
		{35, 19, 35, 45, 80, 73, 46, 48, 81, 34, 0, 9, 85, 9, 71, 55, 40, 27, 41, 48},
		{36, 30, 82, 82, 8, 8, 49, 70, 89, 5, 9, 0, 26, 67, 88, 6, 19, 39, 83, 50},
		{51, 65, 45, 57, 71, 25, 5, 70, 59, 87, 85, 26, 0, 5, 2, 23, 32, 73, 5, 97},
		{1, 74, 24, 45, 39, 71, 78, 65, 34, 8, 9, 67, 5, 0, 72, 94, 14, 36, 49, 83},
		{19, 94, 12, 66, 49, 91, 69, 16, 25, 91, 71, 88, 2, 72, 0, 27, 81, 73, 54, 80},
		{90, 60, 34, 91, 45, 53, 12, 84, 88, 15, 55, 6, 23, 94, 27, 0, 58, 37, 80, 2},
		{14, 55, 86, 14, 19, 99, 18, 50, 62, 46, 40, 19, 32, 14, 81, 58, 0, 70, 49, 49},
		{96, 68, 91, 26, 45, 60, 55, 68, 86, 83, 27, 39, 73, 36, 73, 37, 70, 0, 55, 92},
		{92, 26, 13, 52, 35, 18, 46, 23, 77, 75, 41, 83, 5, 49, 54, 80, 49, 55, 0, 21},
		{25, 49, 76, 68, 1, 83, 63, 54, 68, 27, 48, 50, 97, 83, 80, 2, 49, 92, 21, 0}
	},
	{
		{0, 8, 35, 63, 22, 21, 23, 53, 61, 78, 7, 97, 73, 19, 29, 62, 88, 36, 93, 84},
		{8, 0, 8, 70, 32, 1, 79, 93, 80, 43, 77, 21, 26, 38, 64, 75, 76, 55, 3, 48},
		{35, 8, 0, 6, 12, 60, 59, 12, 1, 27, 81, 45, 63, 3, 87, 98, 52, 25, 35, 6},
		{63, 70, 6, 0, 87, 6, 15, 30, 19, 96, 10, 79, 22, 32, 24, 5, 59, 63, 4, 98},
		{22, 32, 12, 87, 0, 38, 23, 7, 52, 69, 95, 79, 96, 23, 56, 91, 15, 87, 33, 9},
		{21, 1, 60, 6, 38, 0, 34, 64, 22, 80, 3, 44, 99, 66, 12, 61, 3, 68, 48, 57},
		{23, 79, 59, 15, 23, 34, 0, 68, 69, 65, 28, 40, 43, 48, 51, 96, 99, 90, 76, 36},
		{53, 93, 12, 30, 7, 64, 68, 0, 10, 31, 69, 92, 31, 90, 42, 33, 81, 90, 16, 85},
		{61, 80, 1, 19, 52, 22, 69, 10, 0, 62, 20, 90, 35, 21, 48, 64, 24, 35, 68, 72},
		{78, 43, 27, 96, 69, 80, 65, 31, 62, 0, 89, 41, 74, 5, 93, 89, 9, 38, 43, 50},
		{7, 77, 81, 10, 95, 3, 28, 69, 20, 89, 0, 37, 41, 80, 56, 7, 21, 79, 77, 36},
		{97, 21, 45, 79, 79, 44, 40, 92, 90, 41, 37, 0, 82, 31, 17, 18, 4, 65, 65, 25},
		{73, 26, 63, 22, 96, 99, 43, 31, 35, 74, 41, 82, 0, 99, 27, 43, 94, 50, 57, 47},
		{19, 38, 3, 32, 23, 66, 48, 90, 21, 5, 80, 31, 99, 0, 27, 1, 28, 62, 61, 52},
		{29, 64, 87, 24, 56, 12, 51, 42, 48, 93, 56, 17, 27, 27, 0, 56, 30, 29, 70, 9},
		{62, 75, 98, 5, 91, 61, 96, 33, 64, 89, 7, 18, 43, 1, 56, 0, 62, 65, 18, 31},
		{88, 76, 52, 59, 15, 3, 99, 81, 24, 9, 21, 4, 94, 28, 30, 62, 0, 67, 80, 79},
		{36, 55, 25, 63, 87, 68, 90, 90, 35, 38, 79, 65, 50, 62, 29, 65, 67, 0, 69, 41},
		{93, 3, 35, 4, 33, 48, 76, 16, 68, 43, 77, 65, 57, 61, 70, 18, 80, 69, 0, 33},
		{84, 48, 6, 98, 9, 57, 36, 85, 72, 50, 36, 25, 47, 52, 9, 31, 79, 41, 33, 0}
	}
};
