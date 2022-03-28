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
#define POPULATION_SIZE 64

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
 * This transpose distances matrix correspond with KC10-2fl-3rl.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_transposeDistancesMatrix[FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{0, 91, 55, 115, 86, 102, 129, 61, 106, 108},
	{91, 0, 102, 56, 8, 34, 43, 44, 15, 48},
	{55, 102, 0, 96, 94, 93, 145, 58, 115, 92},
	{115, 56, 96, 0, 50, 21, 83, 53, 55, 8},
	{86, 8, 94, 50, 0, 29, 51, 36, 21, 42},
	{102, 34, 93, 21, 29, 0, 65, 41, 36, 13},
	{129, 43, 145, 83, 51, 65, 0, 87, 30, 77},
	{61, 44, 58, 53, 36, 41, 87, 0, 57, 46},
	{106, 15, 115, 55, 21, 36, 30, 57, 0, 49},
	{108, 48, 92, 8, 42, 13, 77, 46, 49, 0}
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
		{0, 2, 14, 182, 0, 0, 1323, 0, 75, 0},
		{2, 0, 4, 0, 4355, 23, 0, 4130, 28, 19},
		{14, 4, 0, 7462, 0, 0, 0, 29, 4653, 15},
		{182, 0, 7462, 0, 1, 0, 0, 5, 2, 0},
		{0, 4355, 0, 1, 0, 17, 2, 16, 9732, 0},
		{0, 23, 0, 0, 17, 0, 0, 6, 23, 0},
		{1323, 0, 0, 0, 2, 0, 0, 2960, 0, 15},
		{0, 4130, 29, 5, 16, 6, 2960, 0, 1931, 3139},
		{75, 28, 4653, 2, 9732, 23, 0, 1931, 0, 3},
		{0, 19, 15, 0, 0, 0, 15, 3139, 3, 0}
	},
	{
		{0, 23, 1, 1, 45, 663, 0, 2746, 0, 5959},
		{23, 0, 204, 145, 0, 0, 4701, 0, 60, 7},
		{1, 204, 0, 0, 3, 69, 559, 0, 0, 37},
		{1, 145, 0, 0, 2181, 2051, 2417, 3888, 62, 1600},
		{45, 0, 3, 2181, 0, 0, 13, 2, 0, 94},
		{663, 0, 69, 2051, 0, 0, 522, 13, 14, 124},
		{0, 4701, 559, 2417, 13, 522, 0, 0, 6580, 1},
		{2746, 0, 0, 3888, 2, 13, 0, 0, 0, 0},
		{0, 60, 0, 62, 0, 14, 6580, 0, 0, 152},
		{5959, 7, 37, 1600, 94, 124, 1, 0, 152, 0}
	}
};
