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
 * This transpose distances matrix correspond with KC10-2fl-1uni.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_transposeDistancesMatrix[FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{0, 61, 50, 80, 25, 31, 42, 100, 19, 64},
	{61, 0, 88, 46, 54, 47, 51, 91, 14, 100},
	{50, 88, 0, 70, 60, 92, 51, 34, 83, 52},
	{80, 46, 70, 0, 65, 15, 47, 45, 41, 79},
	{25, 54, 60, 65, 0, 22, 91, 68, 66, 87},
	{31, 47, 92, 15, 22, 0, 48, 55, 60, 20},
	{42, 51, 51, 47, 91, 48, 0, 25, 71, 51},
	{100, 91, 34, 45, 68, 55, 25, 0, 8, 27},
	{19, 14, 83, 41, 66, 60, 71, 8, 0, 45},
	{64, 100, 52, 79, 87, 20, 51, 27, 45, 0}
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
		{0, 43, 45, 9, 20, 27, 42, 19, 10, 62},
		{43, 0, 80, 71, 9, 25, 39, 32, 19, 90},
		{45, 80, 0, 72, 71, 84, 97, 99, 100, 6},
		{9, 71, 72, 0, 100, 14, 11, 50, 31, 81},
		{20, 9, 71, 100, 0, 56, 11, 7, 26, 96},
		{27, 25, 84, 14, 56, 0, 79, 60, 20, 13},
		{42, 39, 97, 11, 11, 79, 0, 86, 60, 15},
		{19, 32, 99, 50, 7, 60, 86, 0, 42, 31},
		{10, 19, 100, 31, 26, 20, 60, 42, 0, 44},
		{62, 90, 6, 81, 96, 13, 15, 31, 44, 0}
	},
	{
		{0, 34, 17, 2, 71, 91, 36, 31, 29, 97},
		{34, 0, 86, 74, 30, 15, 49, 57, 81, 4},
		{17, 86, 0, 67, 41, 90, 27, 73, 67, 72},
		{2, 74, 67, 0, 44, 73, 40, 6, 18, 61},
		{71, 30, 41, 44, 0, 35, 55, 37, 52, 19},
		{91, 15, 90, 73, 35, 0, 81, 80, 83, 24},
		{36, 49, 27, 40, 55, 81, 0, 87, 9, 37},
		{31, 57, 73, 6, 37, 80, 87, 0, 55, 5},
		{29, 81, 67, 18, 52, 83, 9, 55, 0, 9},
		{97, 4, 72, 61, 19, 24, 37, 5, 9, 0},
	}
};
