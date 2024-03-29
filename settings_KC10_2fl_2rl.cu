/*
 * Executed in a "GeForce RTX 2060M" with
 * (30) Multiprocessors, (64) CUDA Cores/MP.
 */

/*
 * Number of Facilities/locations in the mQAP problem.
 */
#define FACILITIES_LOCATIONS 10

/*
 * Number of objectives in the mQAP problem.
 */
#define OBJECTIVES 2

/*
 * Pt population size. MUST BE A POWER OF TWO
 * due to bitonic sort limitation.
 */
#define POPULATION_SIZE 16

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
 * This transpose distances matrix correspond with KC10-2fl-2rl.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_transposeDistancesMatrix[FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{0, 111, 53, 105, 107, 90, 87, 46, 92, 95},
	{111, 0, 72, 41, 49, 43, 25, 71, 25, 57},
	{53, 72, 0, 87, 55, 74, 57, 49, 65, 87},
	{105, 41, 87, 0, 87, 15, 32, 58, 23, 19},
	{107, 49, 55, 87, 0, 82, 57, 86, 64, 98},
	{90, 43, 74, 15, 82, 0, 25, 43, 18, 16},
	{87, 25, 57, 32, 57, 25, 0, 46, 8, 41},
	{46, 71, 49, 58, 86, 43, 46, 0, 49, 49},
	{92, 25, 65, 23, 64, 18, 8, 49, 0, 34},
	{95, 57, 87, 19, 98, 16, 41, 49, 34, 0}
};

/*
 * Variable with flow matrices in constant device memory.
 * These flow matrices correspond with KC10-2fl-1uni.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_flowMatrices[OBJECTIVES][FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{
		{0, 1, 0, 31, 195, 0, 1156, 7, 1, 7131},
		{1, 0, 8502, 0, 104, 8460, 0, 253, 34, 1116},
		{0, 8502, 0, 8, 48, 0, 88, 809, 5, 21},
		{31, 0, 8, 0, 6, 336, 11, 7273, 1745, 0},
		{195, 104, 48, 6, 0, 979, 0, 162, 188, 5212},
		{0, 8460, 0, 336, 979, 0, 108, 7195, 1, 12},
		{1156, 0, 88, 11, 0, 108, 0, 0, 0, 77},
		{7, 253, 809, 7273, 162, 7195, 0, 0, 0, 0},
		{1, 34, 5, 1745, 188, 1, 0, 0, 0, 118},
		{7131, 1116, 21, 0, 5212, 12, 77, 0, 118, 0}
	},
	{
		{0, 1, 1, 10, 3269, 0, 312, 24, 0, 265},
		{1, 0, 1715, 0, 773, 5756, 0, 2, 16, 295},
		{1, 1715, 0, 10, 216, 0, 37, 9, 0, 0},
		{10, 0, 10, 0, 0, 4, 9, 4348, 999, 0},
		{3269, 773, 216, 0, 0, 5918, 1, 1581, 48, 108},
		{0, 5756, 0, 4, 5918, 0, 24, 691, 0, 1},
		{312, 0, 37, 9, 1, 24, 0, 0, 0, 8},
		{24, 2, 9, 4348, 1581, 691, 0, 0, 0, 6},
		{0, 16, 0, 999, 48, 0, 0, 0, 0, 1513},
		{265, 295, 0, 0, 108, 1, 8, 6, 1513, 0}
	}
};
