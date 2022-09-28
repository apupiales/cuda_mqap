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
		{0, 42, 7, 18, 49, 3, 57, 48, 83, 47, 36, 64, 47, 62, 77, 46, 51, 65, 72, 51},
		{42, 0, 83, 34, 7, 49, 38, 43, 6, 91, 8, 64, 1, 43, 70, 76, 61, 98, 30, 89},
		{7, 83, 0, 49, 89, 65, 11, 9, 75, 39, 30, 13, 9, 97, 80, 76, 7, 51, 37, 14},
		{18, 34, 49, 0, 4, 27, 3, 32, 95, 46, 9, 88, 48, 60, 75, 81, 70, 26, 25, 80},
		{49, 7, 89, 4, 0, 2, 51, 37, 91, 54, 60, 93, 1, 16, 75, 82, 46, 75, 98, 30},
		{3, 49, 65, 27, 2, 0, 85, 16, 40, 41, 32, 80, 60, 69, 12, 59, 73, 48, 14, 55},
		{57, 38, 11, 3, 51, 85, 0, 76, 7, 98, 43, 59, 36, 34, 48, 70, 100, 91, 63, 75},
		{48, 43, 9, 32, 37, 16, 76, 0, 21, 60, 98, 58, 63, 23, 31, 39, 12, 31, 55, 7},
		{83, 6, 75, 95, 91, 40, 7, 21, 0, 91, 84, 99, 37, 96, 83, 25, 64, 79, 44, 5},
		{47, 91, 39, 46, 54, 41, 98, 60, 91, 0, 13, 74, 80, 2, 24, 85, 85, 31, 93, 16},
		{36, 8, 30, 9, 60, 32, 43, 98, 84, 13, 0, 87, 83, 89, 42, 12, 41, 47, 24, 36},
		{64, 64, 13, 88, 93, 80, 59, 58, 99, 74, 87, 0, 22, 93, 37, 62, 6, 100, 82, 35},
		{47, 1, 9, 48, 1, 60, 36, 63, 37, 80, 83, 22, 0, 13, 39, 27, 74, 55, 7, 38},
		{62, 43, 97, 60, 16, 69, 34, 23, 96, 2, 89, 93, 13, 0, 36, 60, 93, 42, 32, 71},
		{77, 70, 80, 75, 75, 12, 48, 31, 83, 24, 42, 37, 39, 36, 0, 48, 75, 53, 35, 28},
		{46, 76, 76, 81, 82, 59, 70, 39, 25, 85, 12, 62, 27, 60, 48, 0, 81, 14, 62, 81},
		{51, 61, 7, 70, 46, 73, 100, 12, 64, 85, 41, 6, 74, 93, 75, 81, 0, 20, 35, 29},
		{65, 98, 51, 26, 75, 48, 91, 31, 79, 31, 47, 100, 55, 42, 53, 14, 20, 0, 48, 86},
		{72, 30, 37, 25, 98, 14, 63, 55, 44, 93, 24, 82, 7, 32, 35, 62, 35, 48, 0, 79},
		{51, 89, 14, 80, 30, 55, 75, 7, 5, 16, 36, 35, 38, 71, 28, 81, 29, 86, 79, 0}
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
		{0, 79, 71, 72, 58, 89, 68, 34, 23, 39, 84, 97, 39, 36, 94, 31, 91, 27, 21, 32},
		{79, 0, 4, 14, 85, 26, 44, 60, 41, 90, 34, 72, 51, 82, 89, 88, 49, 12, 34, 2},
		{71, 4, 0, 19, 9, 69, 30, 97, 83, 68, 31, 46, 83, 69, 33, 24, 80, 72, 61, 2},
		{72, 14, 19, 0, 15, 27, 71, 96, 69, 95, 75, 22, 46, 45, 42, 60, 78, 93, 37, 17},
		{58, 85, 9, 15, 0, 5, 90, 90, 89, 53, 25, 15, 44, 39, 26, 32, 92, 53, 71, 75},
		{89, 26, 69, 27, 5, 0, 42, 38, 83, 63, 49, 3, 61, 11, 67, 82, 34, 59, 44, 94},
		{68, 44, 30, 71, 90, 42, 0, 58, 37, 27, 86, 38, 38, 6, 50, 79, 41, 55, 41, 88},
		{34, 60, 97, 96, 90, 38, 58, 0, 58, 100, 46, 12, 12, 47, 36, 70, 72, 26, 30, 66},
		{23, 41, 83, 69, 89, 83, 37, 58, 0, 58, 78, 87, 41, 52, 40, 34, 46, 63, 29, 95},
		{39, 90, 68, 95, 53, 63, 27, 100, 58, 0, 12, 43, 40, 10, 45, 52, 36, 74, 92, 87},
		{84, 34, 31, 75, 25, 49, 86, 46, 78, 12, 0, 29, 39, 48, 36, 47, 46, 84, 29, 69},
		{97, 72, 46, 22, 15, 3, 38, 12, 87, 43, 29, 0, 67, 2, 11, 89, 32, 64, 39, 47},
		{39, 51, 83, 46, 44, 61, 38, 12, 41, 40, 39, 67, 0, 99, 54, 75, 25, 52, 84, 42},
		{36, 82, 69, 45, 39, 11, 6, 47, 52, 10, 48, 2, 99, 0, 12, 10, 40, 40, 100, 53},
		{94, 89, 33, 42, 26, 67, 50, 36, 40, 45, 36, 11, 54, 12, 0, 24, 30, 48, 71, 4},
		{31, 88, 24, 60, 32, 82, 79, 70, 34, 52, 47, 89, 75, 10, 24, 0, 11, 26, 47, 38},
		{91, 49, 80, 78, 92, 34, 41, 72, 46, 36, 46, 32, 25, 40, 30, 11, 0, 20, 23, 56},
		{27, 12, 72, 93, 53, 59, 55, 26, 63, 74, 84, 64, 52, 40, 48, 26, 20, 0, 25, 42},
		{21, 34, 61, 37, 71, 44, 41, 30, 29, 92, 29, 39, 84, 100, 71, 47, 23, 25, 0, 99},
		{32, 2, 2, 17, 75, 94, 88, 66, 95, 87, 69, 47, 42, 53, 4, 38, 56, 42, 99, 0}
	},
	{
		{0, 72, 42, 80, 68, 86, 34, 52, 1, 42, 82, 88, 65, 11, 91, 29, 97, 42, 19, 21},
		{72, 0, 6, 58, 70, 43, 69, 77, 13, 81, 53, 59, 73, 71, 83, 65, 49, 36, 43, 6},
		{42, 6, 0, 34, 1, 75, 30, 75, 80, 55, 39, 57, 77, 75, 18, 37, 53, 61, 67, 19},
		{80, 58, 34, 0, 26, 54, 91, 64, 65, 99, 84, 5, 36, 38, 37, 41, 92, 86, 2, 32},
		{68, 70, 1, 26, 0, 33, 65, 73, 61, 74, 36, 21, 51, 62, 24, 24, 78, 51, 53, 70},
		{86, 43, 75, 54, 33, 0, 21, 6, 100, 82, 93, 2, 58, 19, 70, 86, 18, 60, 40, 84},
		{34, 69, 30, 91, 65, 21, 0, 54, 42, 49, 76, 59, 23, 37, 41, 75, 42, 22, 27, 97},
		{52, 77, 75, 64, 73, 6, 54, 0, 48, 93, 45, 15, 20, 8, 12, 60, 67, 10, 15, 68},
		{1, 13, 80, 65, 61, 100, 42, 48, 0, 34, 73, 91, 70, 43, 45, 16, 32, 74, 33, 86},
		{42, 81, 55, 99, 74, 82, 49, 93, 34, 0, 47, 60, 51, 23, 35, 47, 27, 73, 73, 68},
		{82, 53, 39, 84, 36, 93, 76, 45, 73, 47, 0, 44, 25, 25, 37, 46, 61, 85, 58, 61},
		{88, 59, 57, 5, 21, 2, 59, 15, 91, 60, 44, 0, 58, 15, 22, 68, 41, 36, 43, 28},
		{65, 73, 77, 36, 51, 58, 23, 20, 70, 51, 25, 58, 0, 83, 43, 64, 16, 69, 94, 50},
		{11, 71, 75, 38, 62, 19, 37, 8, 43, 23, 25, 15, 83, 0, 4, 5, 47, 61, 87, 48},
		{91, 83, 18, 37, 24, 70, 41, 12, 45, 35, 37, 22, 43, 4, 0, 45, 65, 2, 67, 25},
		{29, 65, 37, 41, 24, 86, 75, 60, 16, 47, 46, 68, 64, 5, 45, 0, 30, 49, 57, 54},
		{97, 49, 53, 92, 78, 18, 42, 67, 32, 27, 61, 41, 16, 47, 65, 30, 0, 27, 27, 32},
		{42, 36, 61, 86, 51, 60, 22, 10, 74, 73, 85, 36, 69, 61, 2, 49, 27, 0, 36, 35},
		{19, 43, 67, 2, 53, 40, 27, 15, 33, 73, 58, 43, 94, 87, 67, 57, 27, 36, 0, 81},
		{21, 6, 19, 32, 70, 84, 97, 68, 86, 68, 61, 28, 50, 48, 25, 54, 32, 35, 81, 0}
	}
};
