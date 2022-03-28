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
 * This transpose distances matrix correspond with KC10-2fl-4rl.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_transposeDistancesMatrix[FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{0, 118, 129, 63, 119, 32, 93, 103, 103, 8},
	{118, 0, 55, 76, 2, 98, 31, 51, 79, 122},
	{129, 55, 0, 68, 54, 123, 46, 26, 130, 136},
	{63, 76, 68, 0, 77, 66, 45, 42, 109, 71},
	{119, 2, 54, 77, 0, 100, 31, 50, 81, 124},
	{32, 98, 123, 66, 100, 0, 80, 98, 71, 31},
	{93, 31, 46, 45, 31, 80, 0, 28, 86, 99},
	{103, 51, 26, 42, 50, 98, 28, 0, 114, 110},
	{103, 79, 130, 109, 81, 71, 86, 114, 0, 102},
	{8, 122, 136, 71, 124, 31, 99, 110, 102, 0}
};

/*
 * Variable with flow matrices in constant device memory.
 * These flow matrices correspond with KC10-2fl-4rl.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_flowMatrices[OBJECTIVES][FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{
		{0, 812, 777, 3, 1, 290, 197, 437, 31, 0},
		{812, 0, 0, 0, 0, 0, 0, 4933, 1, 98},
		{777, 0, 0, 0, 160, 166, 0, 0, 26, 7197},
		{3, 0, 0, 0, 0, 202, 41, 2, 2454, 0},
		{1, 0, 160, 0, 0, 0, 0, 97, 9419, 27},
		{290, 0, 166, 202, 0, 0, 0, 150, 175, 326},
		{197, 0, 0, 41, 0, 0, 0, 1008, 0, 0},
		{437, 4933, 0, 2, 97, 150, 1008, 0, 3, 35},
		{31, 1, 26, 2454, 9419, 175, 0, 3, 0, 0},
		{0, 98, 7197, 0, 27, 326, 0, 35, 0, 0}
	},
	{
		{0, 3154, 0, 0, 0, 1267, 156, 0, 0, 30},
		{3154, 0, 1271, 0, 47, 0, 26, 2999, 419, 7039},
		{0, 1271, 0, 0, 0, 118, 2, 1118, 0, 337},
		{0, 0, 0, 0, 0, 247, 19, 12, 7863, 0},
		{0, 47, 0, 0, 0, 0, 0, 0, 85, 0},
		{1267, 0, 118, 247, 0, 0, 2601, 0, 0, 0},
		{156, 26, 2, 19, 0, 2601, 0, 77, 0, 44},
		{0, 2999, 1118, 12, 0, 0, 77, 0, 77, 57},
		{0, 419, 0, 7863, 85, 0, 0, 77, 0, 0},
		{30, 7039, 337, 0, 0, 0, 44, 57, 0, 0}
	}
};
