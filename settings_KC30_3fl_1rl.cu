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
		{0, 144, 52, 39, 76, 97, 59, 82, 110, 75, 138, 128, 43, 45, 101, 120, 44, 58, 44, 94, 165, 137, 35, 71, 75, 101, 74, 91, 99, 83},
		{144, 0, 131, 153, 67, 128, 130, 65, 107, 71, 60, 27, 104, 164, 83, 44, 120, 86, 101, 51, 47, 30, 110, 78, 69, 49, 72, 66, 50, 61},
		{52, 131, 0, 26, 70, 135, 7, 67, 63, 63, 148, 125, 63, 35, 121, 94, 16, 54, 60, 81, 137, 114, 57, 82, 67, 82, 62, 101, 101, 82},
		{39, 153, 26, 0, 87, 134, 32, 88, 89, 82, 161, 143, 68, 11, 129, 119, 33, 69, 66, 102, 162, 138, 60, 93, 85, 104, 81, 113, 116, 98},
		{76, 67, 70, 87, 0, 93, 72, 13, 78, 9, 78, 55, 39, 98, 60, 49, 56, 19, 35, 18, 92, 63, 44, 25, 3, 28, 10, 35, 30, 12},
		{97, 128, 135, 134, 93, 0, 140, 106, 169, 101, 83, 101, 72, 142, 45, 137, 119, 93, 77, 105, 170, 143, 77, 68, 96, 117, 101, 65, 82, 85},
		{59, 130, 7, 32, 72, 140, 0, 67, 56, 63, 150, 126, 68, 40, 125, 92, 21, 56, 64, 81, 133, 112, 63, 85, 68, 80, 63, 104, 102, 84},
		{82, 65, 67, 88, 13, 106, 67, 0, 66, 7, 85, 58, 49, 99, 72, 38, 55, 24, 44, 14, 83, 55, 53, 38, 10, 18, 8, 47, 38, 23},
		{110, 107, 63, 89, 78, 169, 56, 66, 0, 69, 148, 115, 100, 96, 138, 62, 66, 76, 95, 74, 91, 80, 99, 102, 75, 65, 69, 113, 103, 89},
		{75, 71, 63, 82, 9, 101, 63, 7, 69, 0, 87, 62, 41, 93, 69, 46, 49, 16, 36, 20, 90, 62, 45, 33, 5, 26, 1, 45, 39, 22},
		{138, 60, 148, 161, 78, 83, 150, 85, 148, 87, 0, 36, 96, 172, 42, 91, 132, 93, 96, 74, 107, 86, 104, 68, 81, 82, 87, 48, 47, 65},
		{128, 27, 125, 143, 55, 101, 126, 58, 115, 62, 36, 0, 86, 154, 56, 56, 112, 74, 84, 45, 73, 50, 93, 58, 58, 50, 63, 42, 29, 46},
		{43, 104, 63, 68, 39, 72, 68, 49, 100, 41, 96, 86, 0, 78, 61, 87, 47, 26, 5, 57, 131, 102, 8, 28, 39, 67, 40, 48, 56, 42},
		{45, 164, 35, 11, 98, 142, 40, 99, 96, 93, 172, 154, 78, 0, 139, 129, 44, 80, 76, 113, 172, 149, 70, 103, 96, 115, 92, 124, 127, 109},
		{101, 83, 121, 129, 60, 45, 125, 72, 138, 69, 42, 56, 61, 139, 0, 96, 105, 69, 63, 67, 126, 99, 69, 39, 64, 79, 70, 25, 40, 49},
		{120, 44, 94, 119, 49, 137, 92, 38, 62, 46, 91, 56, 87, 129, 96, 0, 86, 63, 83, 32, 44, 20, 91, 72, 48, 20, 47, 71, 56, 52},
		{44, 120, 16, 33, 56, 119, 21, 55, 66, 49, 132, 112, 47, 44, 105, 86, 0, 38, 43, 69, 130, 105, 42, 66, 53, 71, 48, 85, 86, 67},
		{58, 86, 54, 69, 19, 93, 56, 24, 76, 16, 93, 74, 26, 80, 69, 63, 38, 0, 21, 36, 107, 79, 29, 29, 17, 43, 15, 47, 47, 29},
		{44, 101, 60, 66, 35, 77, 64, 44, 95, 36, 96, 84, 5, 76, 63, 83, 43, 21, 0, 53, 127, 98, 9, 27, 35, 62, 36, 48, 55, 39},
		{94, 51, 81, 102, 18, 105, 81, 14, 74, 20, 74, 45, 57, 113, 67, 32, 69, 36, 53, 0, 73, 44, 62, 39, 18, 12, 21, 42, 29, 20},
		{165, 47, 137, 162, 92, 170, 133, 83, 91, 90, 107, 73, 131, 172, 126, 44, 130, 107, 127, 73, 0, 29, 135, 111, 91, 64, 91, 105, 88, 92},
		{137, 30, 114, 138, 63, 143, 112, 55, 80, 62, 86, 50, 102, 149, 99, 20, 105, 79, 98, 44, 29, 0, 107, 82, 63, 36, 63, 77, 60, 63},
		{35, 110, 57, 60, 44, 77, 63, 53, 99, 45, 104, 93, 8, 70, 69, 91, 42, 29, 9, 62, 135, 107, 0, 36, 44, 71, 44, 56, 64, 49},
		{71, 78, 82, 93, 25, 68, 85, 38, 102, 33, 68, 58, 28, 103, 39, 72, 66, 29, 27, 39, 111, 82, 36, 0, 28, 51, 33, 20, 29, 19},
		{75, 69, 67, 85, 3, 96, 68, 10, 75, 5, 81, 58, 39, 96, 64, 48, 53, 17, 35, 18, 91, 63, 44, 28, 0, 27, 6, 39, 33, 16},
		{101, 49, 82, 104, 28, 117, 80, 18, 65, 26, 82, 50, 67, 115, 79, 20, 71, 43, 62, 12, 64, 36, 71, 51, 27, 0, 27, 54, 40, 32},
		{74, 72, 62, 81, 10, 101, 63, 8, 69, 1, 87, 63, 40, 92, 70, 47, 48, 15, 36, 21, 91, 63, 44, 33, 6, 27, 0, 45, 40, 22},
		{91, 66, 101, 113, 35, 65, 104, 47, 113, 45, 48, 42, 48, 124, 25, 71, 85, 47, 48, 42, 105, 77, 56, 20, 39, 54, 45, 0, 17, 23},
		{99, 50, 101, 116, 30, 82, 102, 38, 103, 39, 47, 29, 56, 127, 40, 56, 86, 47, 55, 29, 88, 60, 64, 29, 33, 40, 40, 17, 0, 18},
		{83, 61, 82, 98, 12, 85, 84, 23, 89, 22, 65, 46, 42, 109, 49, 52, 67, 29, 39, 20, 92, 63, 49, 19, 16, 32, 22, 23, 18, 0}

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
		{0, 367, 1261, 931, 38, 1, 5, 0, 3, 2, 0, 10, 10, 1, 1609, 0, 3, 0, 0, 0, 6129, 5807, 0, 4811, 1, 42, 1, 0, 0, 0},
		{367, 0, 0, 1, 11, 2876, 2341, 0, 9681, 832, 102, 5516, 0, 78, 8, 259, 1095, 0, 0, 0, 17, 1729, 0, 0, 2, 8, 0, 30, 0, 5},
		{1261, 0, 0, 1, 3, 4697, 0, 60, 0, 4506, 0, 165, 97, 1, 25, 7, 0, 1800, 16, 0, 0, 8054, 33, 1, 6890, 7590, 19, 594, 2, 4362},
		{931, 1, 1, 0, 5354, 0, 25, 0, 84, 1, 21, 222, 0, 1743, 3, 13, 0, 1, 0, 0, 0, 86, 9, 9417, 3, 0, 13, 0, 0, 6113},
		{38, 11, 3, 5354, 0, 4867, 159, 278, 0, 5699, 411, 206, 0, 0, 10, 102, 0, 187, 14, 16, 0, 137, 0, 0, 165, 1498, 0, 4, 0, 27},
		{1, 2876, 4697, 0, 4867, 0, 0, 0, 6302, 5454, 4, 8, 0, 0, 20, 3, 0, 6333, 0, 9396, 2, 2, 2005, 7, 0, 2, 4683, 3307, 18, 56},
		{5, 2341, 0, 25, 159, 0, 0, 0, 0, 7, 9, 1, 122, 0, 0, 271, 1351, 1344, 0, 0, 536, 0, 0, 0, 77, 0, 502, 0, 4, 77},
		{0, 0, 60, 0, 278, 0, 0, 0, 0, 0, 62, 2, 4275, 52, 0, 4, 60, 2341, 0, 1628, 0, 2877, 0, 7868, 0, 9, 1880, 7336, 0, 0},
		{3, 9681, 0, 84, 0, 6302, 0, 0, 0, 4396, 1, 0, 62, 206, 2319, 162, 469, 0, 8158, 3, 15, 114, 2, 0, 0, 315, 421, 0, 5818, 0},
		{2, 832, 4506, 1, 5699, 5454, 7, 0, 4396, 0, 0, 0, 6, 434, 0, 3, 0, 207, 43, 6013, 26, 45, 1875, 3969, 1927, 0, 4791, 1443, 1036, 0},
		{0, 102, 0, 21, 411, 4, 9, 62, 1, 0, 0, 7, 16, 4, 45, 5, 121, 33, 0, 47, 385, 2487, 0, 0, 91, 0, 51, 0, 2503, 469},
		{10, 5516, 165, 222, 206, 8, 1, 2, 0, 0, 7, 0, 5738, 8, 7247, 0, 10, 597, 901, 1555, 123, 2500, 17, 0, 24, 5367, 2, 535, 3007, 7012},
		{10, 0, 97, 0, 0, 0, 122, 4275, 62, 6, 16, 5738, 0, 185, 7194, 2075, 0, 58, 86, 31, 17, 0, 0, 591, 25, 17, 5559, 0, 208, 1430},
		{1, 78, 1, 1743, 0, 0, 0, 52, 206, 434, 4, 8, 185, 0, 1557, 23, 0, 5, 1, 0, 1, 0, 0, 0, 0, 8, 0, 0, 1378, 2},
		{1609, 8, 25, 3, 10, 20, 0, 0, 2319, 0, 45, 7247, 7194, 1557, 0, 514, 0, 94, 0, 1, 498, 328, 1, 0, 4291, 54, 0, 209, 7981, 27},
		{0, 259, 7, 13, 102, 3, 271, 4, 162, 3, 5, 0, 2075, 23, 514, 0, 0, 0, 1, 0, 499, 3, 1, 191, 13, 531, 4, 0, 161, 0},
		{3, 1095, 0, 0, 0, 0, 1351, 60, 469, 0, 121, 10, 0, 0, 0, 0, 0, 0, 436, 0, 1429, 32, 0, 97, 0, 0, 34, 0, 21, 0},
		{0, 0, 1800, 1, 187, 6333, 1344, 2341, 0, 207, 33, 597, 58, 5, 94, 0, 0, 0, 0, 0, 117, 1, 139, 39, 120, 0, 205, 0, 1235, 863},
		{0, 0, 16, 0, 14, 0, 0, 0, 8158, 43, 0, 901, 86, 1, 0, 1, 436, 0, 0, 0, 0, 499, 0, 0, 65, 9489, 6, 6, 0, 0},
		{0, 0, 0, 0, 16, 9396, 0, 1628, 3, 6013, 47, 1555, 31, 0, 1, 0, 0, 0, 0, 0, 4, 164, 844, 0, 2, 0, 0, 6, 20, 1},
		{6129, 17, 0, 0, 0, 2, 536, 0, 15, 26, 385, 123, 17, 1, 498, 499, 1429, 117, 0, 4, 0, 2010, 0, 0, 1, 1089, 997, 4416, 0, 18},
		{5807, 1729, 8054, 86, 137, 2, 0, 2877, 114, 45, 2487, 2500, 0, 0, 328, 3, 32, 1, 499, 164, 2010, 0, 2643, 0, 1, 0, 5788, 3, 230, 0},
		{0, 0, 33, 9, 0, 2005, 0, 0, 2, 1875, 0, 17, 0, 0, 1, 1, 0, 139, 0, 844, 0, 2643, 0, 4965, 4, 49, 0, 0, 0, 86},
		{4811, 0, 1, 9417, 0, 7, 0, 7868, 0, 3969, 0, 0, 591, 0, 0, 191, 97, 39, 0, 0, 0, 0, 4965, 0, 80, 1, 192, 1, 59, 1716},
		{1, 2, 6890, 3, 165, 0, 77, 0, 0, 1927, 91, 24, 25, 0, 4291, 13, 0, 120, 65, 2, 1, 1, 4, 80, 0, 2, 1, 12, 0, 7753},
		{42, 8, 7590, 0, 1498, 2, 0, 9, 315, 0, 0, 5367, 17, 8, 54, 531, 0, 0, 9489, 0, 1089, 0, 49, 1, 2, 0, 13, 2, 0, 0},
		{1, 0, 19, 13, 0, 4683, 502, 1880, 421, 4791, 51, 2, 5559, 0, 0, 4, 34, 205, 6, 0, 997, 5788, 0, 192, 1, 13, 0, 95, 12, 117},
		{0, 30, 594, 0, 4, 3307, 0, 7336, 0, 1443, 0, 535, 0, 0, 209, 0, 0, 0, 6, 6, 4416, 3, 0, 1, 12, 2, 95, 0, 7037, 149},
		{0, 0, 2, 0, 0, 18, 4, 0, 5818, 1036, 2503, 3007, 208, 1378, 7981, 161, 21, 1235, 0, 20, 0, 230, 0, 59, 0, 0, 12, 7037, 0, 0},
		{0, 5, 4362, 6113, 27, 56, 77, 0, 0, 0, 469, 7012, 1430, 2, 27, 0, 0, 863, 0, 1, 18, 0, 86, 1716, 7753, 0, 117, 149, 0, 0}
	},
	{
		{0, 482, 59, 108, 87, 238, 422, 114, 0, 3, 0, 0, 0, 3, 0, 3, 12, 1, 0, 0, 2, 0, 442, 0, 0, 15, 96, 3, 0, 3},
		{482, 0, 0, 3, 1, 188, 0, 0, 216, 119, 0, 3, 0, 3, 0, 952, 68, 4243, 7, 41, 45, 627, 0, 15, 0, 0, 4, 0, 11, 0},
		{59, 0, 0, 0, 0, 0, 3, 9, 35, 843, 4, 6, 20, 0, 11, 8, 0, 0, 7, 1, 238, 0, 45, 0, 0, 4, 250, 1076, 0, 0},
		{108, 3, 0, 0, 253, 0, 149, 5, 85, 0, 1, 177, 18, 0, 1, 0, 29, 143, 0, 193, 2, 0, 3, 383, 0, 9, 2, 9, 15, 58},
		{87, 1, 0, 253, 0, 1382, 102, 420, 4, 356, 293, 0, 1, 3693, 4, 0, 0, 755, 6, 24, 1, 227, 1674, 1, 985, 0, 3, 0, 4360, 220},
		{238, 188, 0, 0, 1382, 0, 1556, 0, 46, 131, 18, 343, 1906, 4, 0, 62, 1681, 3403, 3, 3684, 0, 3, 322, 2, 9, 1, 22, 54, 10, 0},
		{422, 0, 3, 149, 102, 1556, 0, 0, 1, 2, 0, 0, 0, 2, 2, 1306, 1, 109, 33, 0, 2173, 0, 463, 0, 0, 15, 403, 1035, 161, 69},
		{114, 0, 9, 5, 420, 0, 0, 0, 0, 452, 0, 0, 40, 0, 7, 0, 8, 0, 15, 0, 43, 1, 16, 2278, 151, 0, 3477, 309, 535, 0},
		{0, 216, 35, 85, 4, 46, 1, 0, 0, 2, 3, 4, 0, 0, 1, 39, 1022, 5, 3, 1, 0, 1033, 19, 2, 9, 3, 0, 0, 807, 0},
		{3, 119, 843, 0, 356, 131, 2, 452, 2, 0, 1, 35, 0, 0, 18, 55, 81, 1, 24, 335, 33, 60, 0, 0, 25, 8, 0, 4, 0, 0},
		{0, 0, 4, 1, 293, 18, 0, 0, 3, 1, 0, 0, 36, 0, 0, 3, 0, 19, 2, 4, 89, 48, 11, 108, 759, 5, 3, 28, 20, 45},
		{0, 3, 6, 177, 0, 343, 0, 0, 4, 35, 0, 0, 2032, 0, 2, 1, 0, 0, 0, 1, 5, 0, 6, 24, 0, 1661, 0, 0, 1344, 5143},
		{0, 0, 20, 18, 1, 1906, 0, 40, 0, 0, 36, 2032, 0, 162, 11, 46, 4413, 58, 705, 29, 0, 4, 0, 0, 275, 3, 0, 35, 8, 178},
		{3, 3, 0, 0, 3693, 4, 2, 0, 0, 0, 0, 0, 162, 0, 1553, 503, 4, 0, 78, 1, 0, 0, 101, 0, 4, 0, 3, 575, 8, 0},
		{0, 0, 11, 1, 4, 0, 2, 7, 1, 18, 0, 2, 11, 1553, 0, 3, 1, 0, 129, 42, 1, 392, 0, 2, 0, 505, 5161, 0, 0, 0},
		{3, 952, 8, 0, 0, 62, 1306, 0, 39, 55, 3, 1, 46, 503, 3, 0, 1087, 0, 47, 27, 0, 6, 0, 0, 28, 41, 0, 2, 0, 0},
		{12, 68, 0, 29, 0, 1681, 1, 8, 1022, 81, 0, 0, 4413, 4, 1, 1087, 0, 7, 0, 0, 0, 14, 3, 13, 20, 5, 1, 0, 96, 22},
		{1, 4243, 0, 143, 755, 3403, 109, 0, 5, 1, 19, 0, 58, 0, 0, 0, 7, 0, 50, 349, 0, 0, 7, 29, 7, 1, 0, 0, 0, 2},
		{0, 7, 7, 0, 6, 3, 33, 15, 3, 24, 2, 0, 705, 78, 129, 47, 0, 50, 0, 22, 1969, 69, 115, 2, 1079, 396, 0, 0, 7, 2},
		{0, 41, 1, 193, 24, 3684, 0, 0, 1, 335, 4, 1, 29, 1, 42, 27, 0, 349, 22, 0, 13, 2, 9, 0, 13, 1, 45, 0, 0, 23},
		{2, 45, 238, 2, 1, 0, 2173, 43, 0, 33, 89, 5, 0, 0, 1, 0, 0, 0, 1969, 13, 0, 2, 0, 1, 32, 21, 34, 0, 16, 22},
		{0, 627, 0, 0, 227, 3, 0, 1, 1033, 60, 48, 0, 4, 0, 392, 6, 14, 0, 69, 2, 2, 0, 2359, 14, 2, 3, 0, 50, 16, 0},
		{442, 0, 45, 3, 1674, 322, 463, 16, 19, 0, 11, 6, 0, 101, 0, 0, 3, 7, 115, 9, 0, 2359, 0, 36, 47, 59, 1, 0, 2, 0},
		{0, 15, 0, 383, 1, 2, 0, 2278, 2, 0, 108, 24, 0, 0, 2, 0, 13, 29, 2, 0, 1, 14, 36, 0, 17, 0, 15, 22, 0, 23},
		{0, 0, 0, 0, 985, 9, 0, 151, 9, 25, 759, 0, 275, 4, 0, 28, 20, 7, 1079, 13, 32, 2, 47, 17, 0, 249, 0, 0, 0, 13},
		{15, 0, 4, 9, 0, 1, 15, 0, 3, 8, 5, 1661, 3, 0, 505, 41, 5, 1, 396, 1, 21, 3, 59, 0, 249, 0, 9, 2, 0, 4},
		{96, 4, 250, 2, 3, 22, 403, 3477, 0, 0, 3, 0, 0, 3, 5161, 0, 1, 0, 0, 45, 34, 0, 1, 15, 0, 9, 0, 0, 5, 1112},
		{3, 0, 1076, 9, 0, 54, 1035, 309, 0, 4, 28, 0, 35, 575, 0, 2, 0, 0, 0, 0, 0, 50, 0, 22, 0, 2, 0, 0, 386, 0},
		{0, 11, 0, 15, 4360, 10, 161, 535, 807, 0, 20, 1344, 8, 8, 0, 0, 96, 0, 7, 0, 16, 16, 2, 0, 0, 0, 5, 386, 0, 933},
		{3, 0, 0, 58, 220, 0, 69, 0, 0, 0, 45, 5143, 178, 0, 0, 0, 22, 2, 2, 23, 22, 0, 0, 23, 13, 4, 1112, 0, 933, 0}
	},
	{
		{0, 3, 0, 0, 0, 0, 8, 5524, 83, 1, 4178, 0, 0, 0, 0, 8, 330, 464, 2265, 500, 502, 0, 6931, 0, 0, 0, 0, 0, 23, 217},
		{3, 0, 290, 0, 4, 0, 59, 7, 0, 0, 0, 0, 41, 1, 150, 22, 65, 4, 723, 391, 1839, 0, 0, 24, 0, 0, 1370, 5231, 62, 2264},
		{0, 290, 0, 8, 0, 615, 0, 27, 50, 1, 0, 25, 0, 689, 5, 0, 767, 0, 0, 2536, 0, 2, 0, 0, 1, 0, 2, 2225, 0, 0},
		{0, 0, 8, 0, 0, 2371, 0, 1364, 0, 25, 0, 2, 2, 82, 14, 0, 14, 0, 4, 5, 7, 913, 904, 3773, 42, 12, 6127, 39, 92, 1},
		{0, 4, 0, 0, 0, 3595, 0, 1, 3084, 0, 1, 6, 6, 4334, 0, 0, 0, 0, 0, 6, 0, 0, 60, 580, 386, 1046, 8, 93, 2, 7},
		{0, 0, 615, 2371, 3595, 0, 1, 0, 1363, 0, 0, 1, 0, 5426, 0, 0, 7542, 0, 0, 0, 87, 0, 0, 117, 171, 0, 31, 40, 1, 2},
		{8, 59, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 468, 1, 28, 0, 2815, 380, 0, 7, 7112, 7062, 0, 125, 0, 1, 13, 0},
		{5524, 7, 27, 1364, 1, 0, 0, 0, 2, 0, 11, 33, 13, 2076, 6, 0, 1134, 0, 1712, 7, 3, 0, 6087, 0, 296, 0, 426, 0, 626, 0},
		{83, 0, 50, 0, 3084, 1363, 0, 2, 0, 246, 0, 0, 1, 6, 926, 4817, 0, 206, 0, 511, 8953, 0, 0, 7, 82, 0, 0, 0, 0, 123},
		{1, 0, 1, 25, 0, 0, 5, 0, 246, 0, 510, 414, 54, 0, 2299, 0, 0, 0, 0, 162, 0, 0, 490, 0, 0, 424, 0, 0, 3, 9392},
		{4178, 0, 0, 0, 1, 0, 0, 11, 0, 510, 0, 0, 7, 272, 0, 183, 96, 0, 63, 0, 0, 0, 8186, 19, 132, 20, 0, 443, 0, 0},
		{0, 0, 25, 2, 6, 1, 0, 33, 0, 414, 0, 0, 1082, 160, 0, 0, 150, 0, 996, 0, 98, 3800, 0, 0, 0, 111, 9210, 0, 9, 0},
		{0, 41, 0, 2, 6, 0, 0, 13, 1, 54, 7, 1082, 0, 0, 0, 1268, 2, 0, 53, 0, 0, 4776, 2751, 0, 0, 0, 2, 6558, 0, 26},
		{0, 1, 689, 82, 4334, 5426, 0, 2076, 6, 0, 272, 160, 0, 0, 0, 0, 1699, 0, 0, 19, 4, 1, 8, 1, 556, 81, 2735, 0, 0, 0},
		{0, 150, 5, 14, 0, 0, 468, 6, 926, 2299, 0, 0, 0, 0, 0, 1, 6, 0, 0, 64, 4159, 0, 8479, 116, 0, 0, 5, 0, 0, 0},
		{8, 22, 0, 0, 0, 0, 1, 0, 4817, 0, 183, 0, 1268, 0, 1, 0, 4, 135, 7, 13, 0, 0, 0, 173, 196, 0, 277, 6857, 3, 1067},
		{330, 65, 767, 14, 0, 7542, 28, 1134, 0, 0, 96, 150, 2, 1699, 6, 4, 0, 382, 0, 3699, 0, 0, 3871, 0, 30, 2306, 134, 15, 3894, 613},
		{464, 4, 0, 0, 0, 0, 0, 0, 206, 0, 0, 0, 0, 0, 0, 135, 382, 0, 0, 2482, 0, 0, 241, 0, 0, 25, 1, 619, 0, 6571},
		{2265, 723, 0, 4, 0, 0, 2815, 1712, 0, 0, 63, 996, 53, 0, 0, 7, 0, 0, 0, 1392, 0, 0, 133, 0, 0, 219, 496, 0, 0, 103},
		{500, 391, 2536, 5, 6, 0, 380, 7, 511, 162, 0, 0, 0, 19, 64, 13, 3699, 2482, 1392, 0, 0, 0, 0, 4454, 1155, 0, 0, 0, 0, 713},
		{502, 1839, 0, 7, 0, 87, 0, 3, 8953, 0, 0, 98, 0, 4, 4159, 0, 0, 0, 0, 0, 0, 0, 1810, 556, 8245, 26, 3800, 0, 0, 1},
		{0, 0, 2, 913, 0, 0, 7, 0, 0, 0, 0, 3800, 4776, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 9929, 2112, 0, 0, 2, 22},
		{6931, 0, 0, 904, 60, 0, 7112, 6087, 0, 490, 8186, 0, 2751, 8, 8479, 0, 3871, 241, 133, 0, 1810, 0, 0, 137, 128, 0, 0, 1338, 6, 506},
		{0, 24, 0, 3773, 580, 117, 7062, 0, 7, 0, 19, 0, 0, 1, 116, 173, 0, 0, 0, 4454, 556, 22, 137, 0, 0, 0, 31, 750, 26, 0},
		{0, 0, 1, 42, 386, 171, 0, 296, 82, 0, 132, 0, 0, 556, 0, 196, 30, 0, 0, 1155, 8245, 9929, 128, 0, 0, 7631, 0, 281, 14, 0},
		{0, 0, 0, 12, 1046, 0, 125, 0, 0, 424, 20, 111, 0, 81, 0, 0, 2306, 25, 219, 0, 26, 2112, 0, 0, 7631, 0, 58, 0, 115, 1},
		{0, 1370, 2, 6127, 8, 31, 0, 426, 0, 0, 0, 9210, 2, 2735, 5, 277, 134, 1, 496, 0, 3800, 0, 0, 31, 0, 58, 0, 0, 0, 34},
		{0, 5231, 2225, 39, 93, 40, 1, 0, 0, 0, 443, 0, 6558, 0, 0, 6857, 15, 619, 0, 0, 0, 0, 1338, 750, 281, 0, 0, 0, 5, 4},
		{23, 62, 0, 92, 2, 1, 13, 626, 0, 3, 0, 9, 0, 0, 0, 3, 3894, 0, 0, 0, 0, 2, 6, 26, 14, 115, 0, 5, 0, 2058},
		{217, 2264, 0, 1, 7, 2, 0, 0, 123, 9392, 0, 0, 26, 0, 0, 1067, 613, 6571, 103, 713, 1, 22, 506, 0, 0, 1, 34, 4, 2058, 0}
	}
};
