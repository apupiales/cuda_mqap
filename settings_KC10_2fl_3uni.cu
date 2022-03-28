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
#define POPULATION_SIZE 128

/*
 * Number of times the genetic algorithm is executed.
 */
#define ITERATIONS 25

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
 * This transpose distances matrix correspond with KC10-2fl-3uni.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_transposeDistancesMatrix[FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{0, 85, 44, 50, 84, 50, 60, 71, 22, 95},
	{85, 0, 65, 80, 7, 4, 26, 92, 11, 84},
	{44, 65, 0, 48, 28, 15, 28, 97, 53, 7},
	{50, 80, 48, 0, 91, 34, 59, 50, 87, 47},
	{84, 7, 28, 91, 0, 76, 74, 16, 15, 3},
	{50, 4, 15, 34, 76, 0, 97, 92, 56, 12},
	{60, 26, 28, 59, 74, 97, 0, 82, 78, 16},
	{71, 92, 97, 50, 16, 92, 82, 0, 59, 99},
	{22, 11, 53, 87, 15, 56, 78, 59, 0, 27},
	{95, 84, 7, 47, 3, 12, 16, 99, 27, 0}
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
		{0, 33, 38, 3, 81, 85, 59, 93, 38, 60},
		{33, 0, 61, 26, 63, 85, 85, 74, 88, 81},
		{38, 61, 0, 49, 59, 97, 26, 44, 6, 77},
		{3, 26, 49, 0, 25, 53, 98, 40, 46, 12},
		{81, 63, 59, 25, 0, 68, 14, 27, 4, 67},
		{85, 85, 97, 53, 68, 0, 34, 17, 65, 69},
		{59, 85, 26, 98, 14, 34, 0, 48, 35, 34},
		{93, 74, 44, 40, 27, 17, 48, 0, 10, 89},
		{38, 88, 6, 46, 4, 65, 35, 10, 0, 59},
		{60, 81, 77, 12, 67, 69, 34, 89, 59, 0}
	},
	{
		{0, 78, 68, 88, 1, 16, 49, 1, 62, 38},
		{78, 0, 42, 80, 31, 22, 44, 15, 31, 22},
		{68, 42, 0, 30, 27, 15, 75, 45, 86, 17},
		{88, 80, 30, 0, 87, 35, 8, 62, 58, 82},
		{1, 31, 27, 87, 0, 25, 99, 66, 82, 36},
		{16, 22, 15, 35, 25, 0, 42, 92, 33, 25},
		{49, 44, 75, 8, 99, 42, 0, 56, 53, 89},
		{1, 15, 45, 62, 66, 92, 56, 0, 87, 14},
		{62, 31, 86, 58, 82, 33, 53, 87, 0, 47},
		{38, 22, 17, 82, 36, 25, 89, 14, 47, 0}
	}
};
