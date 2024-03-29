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
#define POPULATION_SIZE 4

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
 * This transpose distances matrix correspond with KC10-2fl-2uni.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_transposeDistancesMatrix[FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{0, 51, 41, 25, 37, 39, 42, 41, 57, 97},
	{51, 0, 55, 67, 40, 87, 70, 87, 23, 28},
	{41, 55, 0, 74, 92, 33, 99, 81, 82, 63},
	{25, 67, 74, 0, 40, 52, 1, 84, 67, 42},
	{37, 40, 92, 40, 0, 67, 72, 31, 99, 43},
	{39, 87, 33, 52, 67, 0, 84, 2, 39, 37},
	{42, 70, 99, 1, 72, 84, 0, 15, 27, 61},
	{41, 87, 81, 84, 31, 2, 15, 0, 91, 36},
	{57, 23, 82, 67, 99, 39, 27, 91, 0, 4},
	{97, 28, 63, 42, 43, 37, 61, 36, 4, 0}
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
		{0, 49, 7, 43, 76, 17, 56, 61, 18, 13},
		{49, 0, 71, 68, 98, 98, 62, 12, 6, 61},
		{7, 71, 0, 100, 34, 43, 22, 98, 43, 79},
		{43, 68, 100, 0, 22, 41, 45, 7, 66, 21},
		{76, 98, 34, 22, 0, 46, 60, 80, 88, 54},
		{17, 98, 43, 41, 46, 0, 90, 57, 92, 9},
		{56, 62, 22, 45, 60, 90, 0, 32, 53, 71},
		{61, 12, 98, 7, 80, 57, 32, 0, 67, 31},
		{18, 6, 43, 66, 88, 92, 53, 67, 0, 90},
		{13, 61, 79, 21, 54, 9, 71, 31, 90, 0}
	},
	{
		{0, 36, 7, 31, 66, 20, 47, 59, 2, 28},
		{36, 0, 72, 71, 94, 97, 69, 14, 9, 42},
		{7, 72, 0, 97, 39, 33, 33, 96, 47, 72},
		{31, 71, 97, 0, 47, 66, 74, 21, 73, 17},
		{66, 94, 39, 47, 0, 51, 59, 67, 80, 58},
		{20, 97, 33, 66, 51, 0, 86, 75, 89, 12},
		{47, 69, 33, 74, 59, 86, 0, 31, 50, 97},
		{59, 14, 96, 21, 67, 75, 31, 0, 59, 31},
		{2, 9, 47, 73, 80, 89, 50, 59, 0, 80},
		{28, 42, 72, 17, 58, 12, 97, 31, 80, 0}
	}
};
