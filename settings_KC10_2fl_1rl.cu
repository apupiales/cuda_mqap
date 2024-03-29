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
 * This transpose distances matrix correspond with KC10-2fl-1rl.dat
 * @see http://www.cs.bham.ac.uk/~jdk/mQAP/
 * @see https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
 */
__constant__ int d_transposeDistancesMatrix[FACILITIES_LOCATIONS][FACILITIES_LOCATIONS] =
{
	{0 ,3 ,60 ,62 ,155 ,29 ,47 ,78 ,83 ,102},
	{3 ,0 ,57 ,61 ,152 ,27 ,44 ,74 ,80 ,99},
	{60 ,57 ,0 ,58 ,95 ,47 ,32 ,19 ,25 ,48},
	{62 ,61 ,58 ,0 ,141 ,33 ,27 ,61 ,61 ,65},
	{155 ,152 ,95 ,141 ,0 ,142 ,123 ,82 ,80 ,80},
	{29 ,27 ,47 ,33 ,142 ,0 ,21 ,60 ,63 ,78},
	{47 ,44 ,32 ,27 ,123 ,21 ,0 ,41 ,43 ,57},
	{78 ,74 ,19 ,61 ,82 ,60 ,41 ,0 ,7 ,31},
	{83 ,80 ,25 ,61 ,80 ,63 ,43 ,7 ,0 ,23},
	{102 ,99 ,48 ,65 ,80 ,78 ,57 ,31 ,23 ,0}
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
		{0, 0, 132, 0, 2558, 667, 0, 572, 1, 200},
		{0, 0, 990, 140, 9445, 1397, 7, 0, 100, 0},
		{132, 990, 0, 7, 40, 2213, 0, 1, 0, 0},
		{0, 140, 7, 0, 58, 0, 1, 0, 4, 70},
		{2558, 9445, 40, 58, 0, 3, 0, 0, 0, 0},
		{667, 1397, 2213, 0, 3, 0, 139, 3, 5169, 101},
		{0, 7, 0, 1, 0, 139, 0, 0, 5659, 0},
		{572, 0, 1, 0, 0, 3, 0, 0, 3388, 1982},
		{1, 100, 0, 4, 0, 5169, 5659, 3388, 0, 1023},
		{200, 0, 0, 70, 0, 101, 0, 1982, 1023, 0}
	},
	{
		{0, 1, 0, 5379, 0, 0, 1, 0, 329, 856},
		{1, 0, 2029, 531, 15, 80, 197, 17, 274, 241},
		{0, 2029, 0, 1605, 0, 194, 0, 2, 4723, 0},
		{5379, 531, 1605, 0, 24, 68, 0, 0, 4847, 2205},
		{0, 15, 0, 24, 0, 1355, 5124, 1610, 0, 0},
		{0, 80, 194, 68, 1355, 0, 549, 0, 151, 2},
		{1, 197, 0, 0, 5124, 549, 0, 5955, 0, 0},
		{0, 17, 2, 0, 1610, 0, 5955, 0, 553, 710},
		{329, 274, 4723, 4847, 0, 151, 0, 553, 0, 758},
		{856, 241, 0, 2205, 0, 2, 0, 710, 758, 0}
	}
};
