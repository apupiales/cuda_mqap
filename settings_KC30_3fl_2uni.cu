/*
 * Executed in a "GeForce RTX 2060M" with
 * (30) Multiprocessors, (64) CUDA Cores/MP.
 */

/*
 * Number of Facilities/locations in the mQAP problem.
 */
#define FACILITIES_LOCATIONS 30

/*
 * Number of objectives in the mQAP problem.
 */
#define OBJECTIVES 3

/*
 * Pt population size. MUST BE A POWER OF TWO
 * due to bitonic sort limitation.
 */
#define POPULATION_SIZE 32

/*
 * Number of times the genetic algorithm is executed.
 */
#define ITERATIONS 70

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
		{0, 44, 33, 37, 83, 34, 75, 1, 75, 41, 43, 71, 1, 75, 46, 36, 39, 67, 84, 37, 29, 96, 39, 63, 51, 45, 68, 70, 90, 88},
		{44, 0, 56, 28, 36, 75, 48, 97, 11, 47, 23, 60, 50, 96, 19, 35, 21, 41, 58, 28, 29, 46, 67, 74, 69, 13, 29, 76, 23, 68},
		{33, 56, 0, 15, 88, 90, 73, 82, 20, 51, 72, 46, 64, 4, 9, 79, 1, 68, 13, 39, 18, 87, 69, 69, 49, 18, 41, 79, 12, 46},
		{37, 28, 15, 0, 91, 36, 84, 65, 2, 36, 73, 80, 14, 11, 36, 65, 35, 44, 18, 24, 41, 84, 80, 55, 61, 84, 13, 50, 13, 31},
		{83, 36, 88, 91, 0, 66, 5, 90, 11, 3, 80, 64, 7, 69, 61, 80, 24, 66, 32, 81, 46, 62, 91, 58, 39, 25, 85, 62, 25, 33},
		{34, 75, 90, 36, 66, 0, 83, 49, 30, 71, 48, 43, 63, 1, 79, 45, 59, 44, 35, 39, 57, 92, 51, 27, 84, 38, 66, 98, 55, 18},
		{75, 48, 73, 84, 5, 83, 0, 3, 57, 97, 13, 82, 50, 52, 67, 62, 16, 3, 71, 19, 58, 2, 43, 71, 78, 74, 24, 64, 75, 78},
		{1, 97, 82, 65, 90, 49, 3, 0, 12, 91, 93, 9, 56, 28, 28, 24, 39, 52, 87, 86, 58, 24, 80, 39, 24, 10, 47, 34, 100, 10},
		{75, 11, 20, 2, 11, 30, 57, 12, 0, 78, 41, 85, 39, 55, 11, 30, 20, 16, 89, 13, 36, 66, 4, 20, 17, 62, 1, 93, 84, 37},
		{41, 47, 51, 36, 3, 71, 97, 91, 78, 0, 69, 54, 77, 82, 37, 85, 83, 7, 62, 72, 42, 93, 25, 32, 26, 58, 30, 40, 16, 57},
		{43, 23, 72, 73, 80, 48, 13, 93, 41, 69, 0, 39, 46, 36, 8, 37, 23, 37, 11, 68, 77, 76, 54, 88, 68, 74, 25, 10, 98, 13},
		{71, 60, 46, 80, 64, 43, 82, 9, 85, 54, 39, 0, 50, 91, 19, 69, 15, 27, 55, 69, 63, 19, 50, 34, 65, 24, 35, 85, 11, 98},
		{1, 50, 64, 14, 7, 63, 50, 56, 39, 77, 46, 50, 0, 16, 69, 95, 86, 71, 52, 73, 60, 91, 77, 16, 20, 33, 49, 33, 36, 25},
		{75, 96, 4, 11, 69, 1, 52, 28, 55, 82, 36, 91, 16, 0, 19, 55, 98, 48, 67, 64, 50, 86, 41, 65, 19, 82, 56, 71, 86, 37},
		{46, 19, 9, 36, 61, 79, 67, 28, 11, 37, 8, 19, 69, 19, 0, 60, 13, 48, 27, 49, 50, 58, 43, 88, 34, 79, 62, 3, 47, 12},
		{36, 35, 79, 65, 80, 45, 62, 24, 30, 85, 37, 69, 95, 55, 60, 0, 71, 85, 42, 69, 66, 19, 48, 70, 13, 85, 18, 35, 51, 96},
		{39, 21, 1, 35, 24, 59, 16, 39, 20, 83, 23, 15, 86, 98, 13, 71, 0, 8, 46, 4, 80, 14, 88, 32, 6, 84, 68, 69, 8, 35},
		{67, 41, 68, 44, 66, 44, 3, 52, 16, 7, 37, 27, 71, 48, 48, 85, 8, 0, 45, 47, 27, 76, 76, 70, 20, 18, 3, 45, 81, 23},
		{84, 58, 13, 18, 32, 35, 71, 87, 89, 62, 11, 55, 52, 67, 27, 42, 46, 45, 0, 41, 100, 99, 23, 40, 41, 95, 43, 81, 37, 15},
		{37, 28, 39, 24, 81, 39, 19, 86, 13, 72, 68, 69, 73, 64, 49, 69, 4, 47, 41, 0, 37, 81, 4, 14, 42, 49, 61, 20, 73, 49},
		{29, 29, 18, 41, 46, 57, 58, 58, 36, 42, 77, 63, 60, 50, 50, 66, 80, 27, 100, 37, 0, 42, 80, 29, 7, 84, 37, 50, 87, 50},
		{96, 46, 87, 84, 62, 92, 2, 24, 66, 93, 76, 19, 91, 86, 58, 19, 14, 76, 99, 81, 42, 0, 73, 14, 15, 17, 4, 39, 16, 64},
		{39, 67, 69, 80, 91, 51, 43, 80, 4, 25, 54, 50, 77, 41, 43, 48, 88, 76, 23, 4, 80, 73, 0, 95, 40, 32, 28, 85, 5, 52},
		{63, 74, 69, 55, 58, 27, 71, 39, 20, 32, 88, 34, 16, 65, 88, 70, 32, 70, 40, 14, 29, 14, 95, 0, 15, 53, 3, 50, 99, 60},
		{51, 69, 49, 61, 39, 84, 78, 24, 17, 26, 68, 65, 20, 19, 34, 13, 6, 20, 41, 42, 7, 15, 40, 15, 0, 5, 31, 55, 89, 48},
		{45, 13, 18, 84, 25, 38, 74, 10, 62, 58, 74, 24, 33, 82, 79, 85, 84, 18, 95, 49, 84, 17, 32, 53, 5, 0, 61, 90, 87, 35},
		{68, 29, 41, 13, 85, 66, 24, 47, 1, 30, 25, 35, 49, 56, 62, 18, 68, 3, 43, 61, 37, 4, 28, 3, 31, 61, 0, 72, 52, 74},
		{70, 76, 79, 50, 62, 98, 64, 34, 93, 40, 10, 85, 33, 71, 3, 35, 69, 45, 81, 20, 50, 39, 85, 50, 55, 90, 72, 0, 29, 67},
		{90, 23, 12, 13, 25, 55, 75, 100, 84, 16, 98, 11, 36, 86, 47, 51, 8, 81, 37, 73, 87, 16, 5, 99, 89, 87, 52, 29, 0, 57},
		{88, 68, 46, 31, 33, 18, 78, 10, 37, 57, 13, 98, 25, 37, 12, 96, 35, 23, 15, 49, 50, 64, 52, 60, 48, 35, 74, 67, 57, 0}
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
		{0, 10, 55, 7, 77, 27, 32, 81, 12, 37, 39, 85, 54, 30, 40, 19, 59, 32, 2, 86, 94, 60, 16, 69, 63, 27, 84, 76, 89, 54},
		{10, 0, 9, 92, 39, 88, 58, 68, 47, 48, 77, 42, 79, 17, 59, 12, 59, 32, 69, 90, 58, 26, 2, 71, 54, 89, 7, 1, 36, 18},
		{55, 9, 0, 2, 17, 71, 50, 36, 20, 97, 98, 31, 28, 68, 45, 61, 27, 29, 61, 73, 67, 91, 22, 9, 63, 37, 85, 16, 14, 20},
		{7, 92, 2, 0, 9, 53, 20, 42, 32, 50, 92, 3, 6, 87, 55, 3, 55, 70, 16, 22, 89, 41, 23, 24, 48, 2, 48, 10, 65, 34},
		{77, 39, 17, 9, 0, 67, 55, 96, 58, 90, 80, 45, 86, 45, 38, 72, 70, 43, 37, 27, 98, 77, 98, 23, 12, 94, 6, 30, 41, 82},
		{27, 88, 71, 53, 67, 0, 23, 16, 5, 82, 35, 12, 24, 21, 93, 94, 79, 53, 94, 60, 65, 40, 48, 79, 32, 91, 6, 51, 72, 3},
		{32, 58, 50, 20, 55, 23, 0, 97, 37, 25, 56, 24, 45, 87, 67, 15, 92, 52, 16, 51, 29, 26, 96, 9, 84, 55, 71, 28, 13, 54},
		{81, 68, 36, 42, 96, 16, 97, 0, 33, 40, 94, 67, 34, 42, 100, 6, 77, 49, 90, 40, 14, 34, 90, 28, 95, 34, 53, 92, 21, 62},
		{12, 47, 20, 32, 58, 5, 37, 33, 0, 18, 42, 76, 10, 11, 78, 57, 8, 22, 38, 11, 67, 97, 40, 23, 78, 96, 19, 42, 68, 34},
		{37, 48, 97, 50, 90, 82, 25, 40, 18, 0, 88, 28, 17, 66, 8, 96, 100, 93, 38, 75, 12, 71, 50, 36, 70, 61, 6, 44, 61, 38},
		{39, 77, 98, 92, 80, 35, 56, 94, 42, 88, 0, 37, 47, 32, 50, 67, 20, 42, 73, 96, 46, 50, 83, 38, 94, 80, 8, 82, 35, 60},
		{85, 42, 31, 3, 45, 12, 24, 67, 76, 28, 37, 0, 32, 89, 81, 41, 95, 13, 87, 84, 46, 47, 55, 5, 53, 87, 11, 34, 21, 95},
		{54, 79, 28, 6, 86, 24, 45, 34, 10, 17, 47, 32, 0, 14, 58, 44, 32, 32, 16, 34, 55, 97, 90, 43, 83, 17, 19, 87, 43, 54},
		{30, 17, 68, 87, 45, 21, 87, 42, 11, 66, 32, 89, 14, 0, 44, 68, 83, 5, 95, 97, 12, 93, 22, 63, 87, 12, 12, 65, 61, 98},
		{40, 59, 45, 55, 38, 93, 67, 100, 78, 8, 50, 81, 58, 44, 0, 42, 67, 68, 62, 5, 29, 35, 53, 90, 44, 26, 74, 71, 53, 19},
		{19, 12, 61, 3, 72, 94, 15, 6, 57, 96, 67, 41, 44, 68, 42, 0, 98, 45, 61, 1, 83, 27, 67, 94, 28, 92, 30, 17, 87, 35},
		{59, 59, 27, 55, 70, 79, 92, 77, 8, 100, 20, 95, 32, 83, 67, 98, 0, 30, 94, 39, 59, 48, 76, 12, 97, 47, 82, 73, 46, 82},
		{32, 32, 29, 70, 43, 53, 52, 49, 22, 93, 42, 13, 32, 5, 68, 45, 30, 0, 18, 88, 55, 60, 21, 50, 90, 49, 63, 21, 84, 8},
		{2, 69, 61, 16, 37, 94, 16, 90, 38, 38, 73, 87, 16, 95, 62, 61, 94, 18, 0, 11, 9, 1, 74, 43, 93, 62, 25, 11, 26, 40},
		{86, 90, 73, 22, 27, 60, 51, 40, 11, 75, 96, 84, 34, 97, 5, 1, 39, 88, 11, 0, 58, 51, 18, 74, 100, 65, 72, 13, 100, 39},
		{94, 58, 67, 89, 98, 65, 29, 14, 67, 12, 46, 46, 55, 12, 29, 83, 59, 55, 9, 58, 0, 42, 12, 29, 62, 53, 63, 9, 78, 58},
		{60, 26, 91, 41, 77, 40, 26, 34, 97, 71, 50, 47, 97, 93, 35, 27, 48, 60, 1, 51, 42, 0, 38, 27, 25, 61, 28, 35, 79, 36},
		{16, 2, 22, 23, 98, 48, 96, 90, 40, 50, 83, 55, 90, 22, 53, 67, 76, 21, 74, 18, 12, 38, 0, 52, 30, 100, 81, 31, 22, 67},
		{69, 71, 9, 24, 23, 79, 9, 28, 23, 36, 38, 5, 43, 63, 90, 94, 12, 50, 43, 74, 29, 27, 52, 0, 72, 67, 47, 4, 84, 5},
		{63, 54, 63, 48, 12, 32, 84, 95, 78, 70, 94, 53, 83, 87, 44, 28, 97, 90, 93, 100, 62, 25, 30, 72, 0, 72, 97, 53, 78, 38},
		{27, 89, 37, 2, 94, 91, 55, 34, 96, 61, 80, 87, 17, 12, 26, 92, 47, 49, 62, 65, 53, 61, 100, 67, 72, 0, 59, 58, 83, 86},
		{84, 7, 85, 48, 6, 6, 71, 53, 19, 6, 8, 11, 19, 12, 74, 30, 82, 63, 25, 72, 63, 28, 81, 47, 97, 59, 0, 20, 43, 24},
		{76, 1, 16, 10, 30, 51, 28, 92, 42, 44, 82, 34, 87, 65, 71, 17, 73, 21, 11, 13, 9, 35, 31, 4, 53, 58, 20, 0, 61, 70},
		{89, 36, 14, 65, 41, 72, 13, 21, 68, 61, 35, 21, 43, 61, 53, 87, 46, 84, 26, 100, 78, 79, 22, 84, 78, 83, 43, 61, 0, 77},
		{54, 18, 20, 34, 82, 3, 54, 62, 34, 38, 60, 95, 54, 98, 19, 35, 82, 8, 40, 39, 58, 36, 67, 5, 38, 86, 24, 70, 77, 0}
	},
	{
		{0, 30, 34, 11, 51, 23, 90, 42, 13, 68, 31, 81, 69, 90, 19, 8, 18, 36, 14, 94, 62, 77, 12, 67, 14, 71, 53, 30, 53, 82},
		{30, 0, 55, 54, 50, 44, 49, 9, 75, 92, 79, 41, 63, 8, 73, 10, 79, 93, 41, 73, 44, 43, 58, 84, 17, 74, 21, 65, 18, 48},
		{34, 55, 0, 14, 29, 26, 22, 29, 15, 46, 60, 66, 1, 79, 100, 30, 52, 17, 11, 46, 88, 66, 42, 28, 18, 60, 53, 51, 44, 15},
		{11, 54, 14, 0, 8, 98, 31, 7, 93, 66, 92, 75, 40, 78, 57, 37, 81, 51, 29, 89, 80, 53, 45, 81, 88, 45, 47, 17, 57, 35},
		{51, 50, 29, 8, 0, 80, 62, 61, 3, 49, 87, 50, 52, 20, 45, 30, 72, 64, 42, 35, 88, 98, 98, 46, 12, 76, 30, 38, 34, 71},
		{23, 44, 26, 98, 80, 0, 47, 1, 43, 88, 8, 33, 50, 1, 73, 72, 55, 77, 53, 90, 26, 27, 99, 54, 2, 63, 10, 86, 42, 32},
		{90, 49, 22, 31, 62, 47, 0, 54, 32, 60, 84, 21, 47, 56, 75, 21, 72, 73, 27, 24, 9, 5, 99, 27, 59, 32, 18, 46, 3, 99},
		{42, 9, 29, 7, 61, 1, 54, 0, 85, 60, 94, 47, 40, 39, 96, 81, 64, 43, 62, 3, 19, 43, 35, 55, 29, 13, 80, 57, 21, 32},
		{13, 75, 15, 93, 3, 43, 32, 85, 0, 20, 12, 70, 4, 11, 5, 75, 15, 25, 47, 60, 95, 51, 43, 78, 73, 62, 50, 68, 1, 35},
		{68, 92, 46, 66, 49, 88, 60, 60, 20, 0, 75, 30, 63, 2, 30, 89, 27, 94, 58, 66, 14, 80, 29, 39, 29, 61, 60, 47, 32, 65},
		{31, 79, 60, 92, 87, 8, 84, 94, 12, 75, 0, 40, 28, 95, 74, 17, 24, 20, 77, 37, 76, 95, 68, 36, 84, 89, 24, 75, 53, 39},
		{81, 41, 66, 75, 50, 33, 21, 47, 70, 30, 40, 0, 58, 25, 19, 24, 25, 43, 60, 63, 16, 66, 80, 28, 98, 75, 10, 25, 11, 58},
		{69, 63, 1, 40, 52, 50, 47, 40, 4, 63, 28, 58, 0, 31, 59, 38, 63, 57, 28, 96, 38, 39, 67, 25, 87, 41, 35, 98, 18, 52},
		{90, 8, 79, 78, 20, 1, 56, 39, 11, 2, 95, 25, 31, 0, 39, 94, 69, 10, 67, 84, 38, 74, 26, 37, 73, 51, 36, 97, 9, 66},
		{19, 73, 100, 57, 45, 73, 75, 96, 5, 30, 74, 19, 59, 39, 0, 26, 49, 66, 71, 30, 32, 70, 86, 43, 24, 48, 51, 14, 88, 40},
		{8, 10, 30, 37, 30, 72, 21, 81, 75, 89, 17, 24, 38, 94, 26, 0, 86, 69, 22, 9, 98, 62, 5, 45, 90, 55, 2, 39, 66, 26},
		{18, 79, 52, 81, 72, 55, 72, 64, 15, 27, 24, 25, 63, 69, 49, 86, 0, 48, 72, 87, 51, 35, 44, 52, 90, 59, 57, 64, 62, 82},
		{36, 93, 17, 51, 64, 77, 73, 43, 25, 94, 20, 43, 57, 10, 66, 69, 48, 0, 33, 76, 65, 72, 21, 32, 55, 62, 98, 28, 76, 3},
		{14, 41, 11, 29, 42, 53, 27, 62, 47, 58, 77, 60, 28, 67, 71, 22, 72, 33, 0, 42, 46, 12, 66, 61, 84, 52, 75, 100, 42, 63},
		{94, 73, 46, 89, 35, 90, 24, 3, 60, 66, 37, 63, 96, 84, 30, 9, 87, 76, 42, 0, 18, 52, 30, 56, 63, 91, 46, 46, 62, 50},
		{62, 44, 88, 80, 88, 26, 9, 19, 95, 14, 76, 16, 38, 38, 32, 98, 51, 65, 46, 18, 0, 75, 25, 26, 50, 48, 21, 23, 92, 63},
		{77, 43, 66, 53, 98, 27, 5, 43, 51, 80, 95, 66, 39, 74, 70, 62, 35, 72, 12, 52, 75, 0, 49, 44, 31, 40, 15, 69, 13, 70},
		{12, 58, 42, 45, 98, 99, 99, 35, 43, 29, 68, 80, 67, 26, 86, 5, 44, 21, 66, 30, 25, 49, 0, 5, 17, 65, 93, 44, 66, 42},
		{67, 84, 28, 81, 46, 54, 27, 55, 78, 39, 36, 28, 25, 37, 43, 45, 52, 32, 61, 56, 26, 44, 5, 0, 61, 98, 66, 48, 85, 10},
		{14, 17, 18, 88, 12, 2, 59, 29, 73, 29, 84, 98, 87, 73, 24, 90, 90, 55, 84, 63, 50, 31, 17, 61, 0, 31, 57, 5, 33, 86},
		{71, 74, 60, 45, 76, 63, 32, 13, 62, 61, 89, 75, 41, 51, 48, 55, 59, 62, 52, 91, 48, 40, 65, 98, 31, 0, 66, 99, 1, 90},
		{53, 21, 53, 47, 30, 10, 18, 80, 50, 60, 24, 10, 35, 36, 51, 2, 57, 98, 75, 46, 21, 15, 93, 66, 57, 66, 0, 4, 59, 76},
		{30, 65, 51, 17, 38, 86, 46, 57, 68, 47, 75, 25, 98, 97, 14, 39, 64, 28, 100, 46, 23, 69, 44, 48, 5, 99, 4, 0, 34, 41},
		{53, 18, 44, 57, 34, 42, 3, 21, 1, 32, 53, 11, 18, 9, 88, 66, 62, 76, 42, 62, 92, 13, 66, 85, 33, 1, 59, 34, 0, 60},
		{82, 48, 15, 35, 71, 32, 99, 32, 35, 65, 39, 58, 52, 66, 40, 26, 82, 3, 63, 50, 63, 70, 42, 10, 86, 90, 76, 41, 60, 0}
	},
	{
		{0, 13, 6, 35, 50, 56, 31, 68, 25, 48, 31, 85, 81, 42, 27, 45, 26, 5, 20, 37, 60, 66, 62, 69, 25, 5, 96, 77, 81, 96},
		{13, 0, 45, 65, 53, 74, 69, 77, 92, 80, 91, 15, 34, 34, 45, 22, 51, 52, 12, 41, 55, 23, 26, 45, 71, 24, 55, 33, 23, 49},
		{6, 45, 0, 16, 22, 41, 70, 76, 9, 61, 65, 83, 25, 95, 4, 45, 6, 49, 90, 96, 41, 78, 29, 30, 45, 1, 39, 12, 41, 22},
		{35, 65, 16, 0, 29, 8, 29, 36, 3, 82, 35, 46, 4, 87, 30, 58, 84, 13, 86, 6, 41, 39, 41, 23, 40, 24, 64, 82, 97, 12},
		{50, 53, 22, 29, 0, 87, 86, 84, 83, 83, 62, 45, 67, 85, 47, 79, 58, 23, 50, 49, 38, 83, 74, 33, 67, 62, 46, 78, 55, 46},
		{56, 74, 41, 8, 87, 0, 45, 79, 12, 56, 38, 34, 28, 46, 67, 72, 90, 37, 50, 77, 70, 21, 46, 65, 56, 57, 19, 38, 17, 40},
		{31, 69, 70, 29, 86, 45, 0, 48, 18, 57, 88, 41, 3, 20, 85, 20, 92, 86, 22, 40, 4, 43, 95, 39, 86, 8, 69, 54, 10, 57},
		{68, 77, 76, 36, 84, 79, 48, 0, 70, 40, 40, 39, 36, 45, 82, 2, 67, 78, 89, 8, 15, 49, 75, 22, 72, 89, 51, 36, 25, 43},
		{25, 92, 9, 3, 83, 12, 18, 70, 0, 27, 52, 71, 46, 1, 28, 85, 27, 2, 60, 6, 68, 42, 27, 14, 56, 84, 13, 100, 64, 44},
		{48, 80, 61, 82, 83, 56, 57, 40, 27, 0, 60, 53, 63, 82, 51, 77, 43, 90, 85, 94, 31, 58, 29, 46, 98, 59, 48, 46, 77, 37},
		{31, 91, 65, 35, 62, 38, 88, 40, 52, 60, 0, 63, 19, 9, 16, 99, 2, 8, 83, 96, 92, 80, 91, 5, 85, 51, 52, 99, 89, 51},
		{85, 15, 83, 46, 45, 34, 41, 39, 71, 53, 63, 0, 11, 89, 11, 57, 92, 43, 41, 20, 36, 66, 55, 72, 78, 35, 17, 46, 2, 88},
		{81, 34, 25, 4, 67, 28, 3, 36, 46, 63, 19, 11, 0, 5, 75, 13, 6, 31, 35, 69, 86, 68, 15, 41, 58, 49, 35, 78, 42, 46},
		{42, 34, 95, 87, 85, 46, 20, 45, 1, 82, 9, 89, 5, 0, 76, 81, 31, 12, 88, 87, 1, 90, 46, 93, 93, 47, 56, 92, 45, 69},
		{27, 45, 4, 30, 47, 67, 85, 82, 28, 51, 16, 11, 75, 76, 0, 62, 93, 70, 75, 15, 48, 9, 99, 32, 47, 70, 58, 50, 97, 4},
		{45, 22, 45, 58, 79, 72, 20, 2, 85, 77, 99, 57, 13, 81, 62, 0, 67, 10, 84, 1, 79, 21, 88, 89, 77, 78, 60, 60, 46, 11},
		{26, 51, 6, 84, 58, 90, 92, 67, 27, 43, 2, 92, 6, 31, 93, 67, 0, 4, 87, 49, 14, 76, 25, 13, 60, 13, 91, 69, 88, 55},
		{5, 52, 49, 13, 23, 37, 86, 78, 2, 90, 8, 43, 31, 12, 70, 10, 4, 0, 42, 95, 44, 74, 6, 85, 66, 52, 48, 23, 84, 36},
		{20, 12, 90, 86, 50, 50, 22, 89, 60, 85, 83, 41, 35, 88, 75, 84, 87, 42, 0, 28, 55, 15, 57, 24, 71, 62, 39, 7, 39, 40},
		{37, 41, 96, 6, 49, 77, 40, 8, 6, 94, 96, 20, 69, 87, 15, 1, 49, 95, 28, 0, 15, 26, 13, 58, 83, 28, 90, 36, 68, 36},
		{60, 55, 41, 41, 38, 70, 4, 15, 68, 31, 92, 36, 86, 1, 48, 79, 14, 44, 55, 15, 0, 71, 75, 38, 94, 62, 65, 6, 75, 70},
		{66, 23, 78, 39, 83, 21, 43, 49, 42, 58, 80, 66, 68, 90, 9, 21, 76, 74, 15, 26, 71, 0, 23, 17, 66, 27, 16, 80, 64, 58},
		{62, 26, 29, 41, 74, 46, 95, 75, 27, 29, 91, 55, 15, 46, 99, 88, 25, 6, 57, 13, 75, 23, 0, 16, 17, 97, 11, 26, 64, 94},
		{69, 45, 30, 23, 33, 65, 39, 22, 14, 46, 5, 72, 41, 93, 32, 89, 13, 85, 24, 58, 38, 17, 16, 0, 28, 51, 8, 62, 47, 57},
		{25, 71, 45, 40, 67, 56, 86, 72, 56, 98, 85, 78, 58, 93, 47, 77, 60, 66, 71, 83, 94, 66, 17, 28, 0, 70, 93, 42, 58, 25},
		{5, 24, 1, 24, 62, 57, 8, 89, 84, 59, 51, 35, 49, 47, 70, 78, 13, 52, 62, 28, 62, 27, 97, 51, 70, 0, 49, 75, 60, 65},
		{96, 55, 39, 64, 46, 19, 69, 51, 13, 48, 52, 17, 35, 56, 58, 60, 91, 48, 39, 90, 65, 16, 11, 8, 93, 49, 0, 16, 90, 28},
		{77, 33, 12, 82, 78, 38, 54, 36, 100, 46, 99, 46, 78, 92, 50, 60, 69, 23, 7, 36, 6, 80, 26, 62, 42, 75, 16, 0, 75, 69},
		{81, 23, 41, 97, 55, 17, 10, 25, 64, 77, 89, 2, 42, 45, 97, 46, 88, 84, 39, 68, 75, 64, 64, 47, 58, 60, 90, 75, 0, 85},
		{96, 49, 22, 12, 46, 40, 57, 43, 44, 37, 51, 88, 46, 69, 4, 11, 55, 36, 40, 36, 70, 58, 94, 57, 25, 65, 28, 69, 85, 0}
	}
};
