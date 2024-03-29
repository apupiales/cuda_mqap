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
 * This transpose distances matrix correspond with KC10-2fl-5rl.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_transposeDistancesMatrix[FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{0, 60, 17, 75, 34, 53, 16, 79, 48, 93},
	{60, 0, 67, 101, 26, 67, 74, 76, 76, 86},
	{17, 67, 0, 91, 44, 41, 11, 96, 65, 110},
	{75, 101, 91, 0, 82, 128, 84, 41, 27, 49},
	{34, 26, 44, 82, 0, 58, 50, 67, 55, 80},
	{53, 67, 41, 128, 58, 0, 51, 124, 101, 138},
	{16, 74, 11, 84, 50, 51, 0, 93, 60, 108},
	{79, 76, 96, 41, 67, 124, 93, 0, 39, 14},
	{48, 76, 65, 27, 55, 101, 60, 39, 0, 53},
	{93, 86, 110, 49, 80, 138, 108, 14, 53, 0}
};

/*
 * Variable with flow matrices in constant device memory.
 * These flow matrices correspond with KC10-2fl-5rl.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_flowMatrices[OBJECTIVES][FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 14404},
		{0, 0, 4, 37032, 0, 2318, 0, 0, 0, 0},
		{0, 4, 0, 6390, 0, 0, 0, 0, 0, 0},
		{0, 37032, 6390, 0, 127, 0, 0, 13, 0, 1},
		{0, 0, 0, 127, 0, 1602, 1, 0, 0, 23},
		{0, 2318, 0, 0, 1602, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 1, 0, 0, 10, 0, 0},
		{0, 0, 0, 13, 0, 0, 10, 0, 0, 7617},
		{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		{14404, 0, 0, 1, 23, 0, 0, 7617, 0, 0},
	},
	{
		{0, 72306, 0, 0, 12, 0, 1005, 22, 0, 0},
		{72306, 0, 0, 0, 5, 51, 1, 1, 0, 2},
		{0, 0, 0, 2503, 0, 2, 0, 26, 0, 0},
		{0, 0, 2503, 0, 0, 276, 0, 0, 0, 39},
		{12, 5, 0, 0, 0, 0, 0, 100, 0, 0},
		{0, 51, 2, 276, 0, 0, 15131, 0, 0, 89},
		{1005, 1, 0, 0, 0, 15131, 0, 0, 0, 0},
		{22, 1, 26, 0, 100, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 2, 0, 39, 0, 89, 0, 0, 0, 0}
	}
};
