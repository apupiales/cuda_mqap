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
		{0, 66, 47, 93, 101, 33, 61, 71, 5, 9, 42, 51, 33, 102, 8, 92, 24, 91, 82, 45},
		{66, 0, 84, 71, 55, 48, 116, 132, 70, 61, 76, 23, 99, 41, 60, 98, 42, 78, 39, 90},
		{47, 84, 0, 134, 133, 75, 37, 57, 50, 55, 7, 82, 58, 105, 52, 51, 50, 135, 115, 10},
		{93, 71, 134, 0, 34, 60, 154, 163, 93, 84, 127, 55, 115, 102, 85, 163, 84, 12, 33, 135},
		{101, 55, 133, 34, 0, 70, 161, 173, 103, 93, 126, 51, 130, 75, 93, 153, 85, 46, 18, 137},
		{33, 48, 75, 60, 70, 0, 94, 104, 33, 24, 69, 26, 60, 89, 25, 112, 27, 60, 51, 75},
		{61, 116, 37, 154, 161, 94, 0, 21, 62, 70, 43, 109, 50, 142, 69, 78, 76, 153, 142, 28},
		{71, 132, 57, 163, 173, 104, 21, 0, 70, 80, 62, 122, 50, 160, 79, 99, 90, 160, 154, 47},
		{5, 70, 50, 93, 103, 33, 62, 70, 0, 9, 47, 54, 29, 107, 10, 97, 29, 91, 84, 47},
		{9, 61, 55, 84, 93, 24, 70, 80, 9, 0, 50, 44, 38, 100, 2, 99, 22, 82, 74, 54},
		{42, 76, 7, 127, 126, 69, 43, 62, 47, 50, 0, 75, 58, 99, 48, 51, 43, 129, 108, 15},
		{51, 23, 82, 55, 51, 26, 109, 122, 54, 44, 75, 0, 83, 64, 44, 108, 33, 60, 33, 85},
		{33, 99, 58, 115, 130, 60, 50, 50, 29, 38, 58, 83, 0, 135, 39, 109, 57, 111, 112, 50},
		{102, 41, 105, 102, 75, 89, 142, 160, 107, 100, 99, 64, 135, 0, 98, 99, 78, 112, 68, 114},
		{8, 60, 52, 85, 93, 25, 69, 79, 10, 2, 48, 44, 39, 98, 0, 96, 20, 84, 75, 52},
		{92, 98, 51, 163, 153, 112, 78, 99, 97, 99, 51, 108, 109, 99, 96, 0, 84, 168, 137, 59},
		{24, 42, 50, 84, 85, 27, 76, 90, 29, 22, 43, 33, 57, 78, 20, 84, 0, 86, 66, 52},
		{91, 78, 135, 12, 46, 60, 153, 160, 91, 82, 129, 60, 111, 112, 84, 168, 86, 0, 44, 135},
		{82, 39, 115, 33, 18, 51, 142, 154, 84, 74, 108, 33, 112, 68, 75, 137, 66, 44, 0, 119},
		{45, 90, 10, 135, 137, 75, 28, 47, 47, 54, 15, 85, 50, 114, 52, 59, 52, 135, 119, 0}
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
		{0, 0, 1274, 4727, 0, 1934, 594, 0, 1619, 951, 1, 4537, 0, 0, 4211, 21, 1, 9, 5, 0},
		{0, 0, 5090, 0, 0, 130, 5, 0, 12, 2787, 7560, 52, 9, 6, 7556, 15, 6, 0, 7346, 3},
		{1274, 5090, 0, 518, 3, 0, 10, 0, 99, 4, 3, 2, 0, 0, 0, 1690, 4665, 0, 113, 3},
		{4727, 0, 518, 0, 0, 1, 688, 23, 0, 2955, 0, 34, 677, 1362, 1, 635, 18, 0, 9644, 0},
		{0, 0, 3, 0, 0, 1729, 96, 455, 2, 2, 1726, 8, 51, 2, 970, 1630, 0, 71, 195, 0},
		{1934, 130, 0, 1, 1729, 0, 117, 0, 94, 1, 107, 7937, 8998, 0, 86, 12, 0, 47, 31, 153},
		{594, 5, 10, 688, 96, 117, 0, 45, 220, 2, 0, 1, 8707, 0, 0, 1, 38, 138, 1, 0},
		{0, 0, 0, 23, 455, 0, 45, 0, 33, 13, 0, 0, 2, 12, 0, 7, 1251, 1, 0, 0},
		{1619, 12, 99, 0, 2, 94, 220, 33, 0, 0, 340, 6944, 0, 0, 0, 0, 0, 120, 0, 74},
		{951, 2787, 4, 2955, 2, 1, 2, 13, 0, 0, 0, 7695, 5965, 0, 478, 8592, 417, 0, 7, 0},
		{1, 7560, 3, 0, 1726, 107, 0, 0, 340, 0, 0, 0, 0, 0, 10, 0, 7, 671, 0, 5406},
		{4537, 52, 2, 34, 8, 7937, 1, 0, 6944, 7695, 0, 0, 4, 0, 56, 0, 20, 22, 208, 0},
		{0, 9, 0, 677, 51, 8998, 8707, 2, 0, 5965, 0, 4, 0, 1126, 1338, 0, 0, 0, 0, 1},
		{0, 6, 0, 1362, 2, 0, 0, 12, 0, 0, 0, 0, 1126, 0, 11, 27, 1399, 47, 5004, 550},
		{4211, 7556, 0, 1, 970, 86, 0, 0, 0, 478, 10, 56, 1338, 11, 0, 0, 38, 636, 251, 3944},
		{21, 15, 1690, 635, 1630, 12, 1, 7, 0, 8592, 0, 0, 0, 27, 0, 0, 1, 0, 0, 0},
		{1, 6, 4665, 18, 0, 0, 38, 1251, 0, 417, 7, 20, 0, 1399, 38, 1, 0, 6523, 0, 6},
		{9, 0, 0, 0, 71, 47, 138, 1, 120, 0, 671, 22, 0, 47, 636, 0, 6523, 0, 0, 476},
		{5, 7346, 113, 9644, 195, 31, 1, 0, 0, 7, 0, 208, 0, 5004, 251, 0, 0, 0, 0, 44},
		{0, 3, 3, 0, 0, 153, 0, 0, 74, 0, 5406, 0, 1, 550, 3944, 0, 6, 476, 44, 0}
	},
	{
		{0, 0, 2501, 7345, 0, 69, 1652, 1, 165, 1, 3, 2, 23, 0, 4, 74, 112, 7, 12, 113},
		{0, 0, 1627, 18, 1, 170, 29, 39, 118, 84, 2535, 83, 3, 59, 10, 1, 80, 0, 5, 39},
		{2501, 1627, 0, 0, 11, 25, 0, 0, 24, 386, 302, 18, 0, 0, 49, 88, 3, 1, 8, 35},
		{7345, 18, 0, 0, 0, 0, 75, 436, 52, 681, 22, 988, 1528, 13, 1, 101, 0, 2, 1764, 12},
		{0, 1, 11, 0, 0, 51, 203, 12, 1, 63, 4888, 1, 2, 3, 108, 1358, 0, 123, 1, 46},
		{69, 170, 25, 0, 51, 0, 3, 0, 1, 32, 9, 13, 5, 172, 2, 2, 13, 2, 0, 87},
		{1652, 29, 0, 75, 203, 3, 0, 0, 58, 0, 0, 134, 58, 34, 0, 1, 44, 3, 0, 0},
		{1, 39, 0, 436, 12, 0, 0, 0, 29, 226, 53, 34, 0, 132, 0, 1, 5, 271, 0, 4},
		{165, 118, 24, 52, 1, 1, 58, 29, 0, 0, 4, 7846, 0, 0, 18, 0, 0, 13, 0, 24},
		{1, 84, 386, 681, 63, 32, 0, 226, 0, 0, 0, 27, 2, 0, 1355, 2557, 30, 0, 39, 14},
		{3, 2535, 302, 22, 4888, 9, 0, 53, 4, 0, 0, 0, 0, 2, 3, 0, 16, 14, 5, 32},
		{2, 83, 18, 988, 1, 13, 134, 34, 7846, 27, 0, 0, 48, 0, 1, 26, 0, 799, 12, 0},
		{23, 3, 0, 1528, 2, 5, 58, 0, 0, 2, 0, 48, 0, 1, 4290, 28, 15, 7, 0, 2},
		{0, 59, 0, 13, 3, 172, 34, 132, 0, 0, 2, 0, 1, 0, 65, 239, 1708, 37, 11, 7},
		{4, 10, 49, 1, 108, 2, 0, 0, 18, 1355, 3, 1, 4290, 65, 0, 26, 0, 824, 63, 2},
		{74, 1, 88, 101, 1358, 2, 1, 1, 0, 2557, 0, 26, 28, 239, 26, 0, 17, 52, 59, 0},
		{112, 80, 3, 0, 0, 13, 44, 5, 0, 30, 16, 0, 15, 1708, 0, 17, 0, 3, 5, 23},
		{7, 0, 1, 2, 123, 2, 3, 271, 13, 0, 14, 799, 7, 37, 824, 52, 3, 0, 7, 66},
		{12, 5, 8, 1764, 1, 0, 0, 0, 0, 39, 5, 12, 0, 11, 63, 59, 5, 7, 0, 0},
		{113, 39, 35, 12, 46, 87, 0, 4, 24, 14, 32, 0, 2, 7, 2, 0, 23, 66, 0, 0}
	}
};
