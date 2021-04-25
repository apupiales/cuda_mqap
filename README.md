# (cuda_mqap) Parallel Implementation of NSGA-2 + Addapted Greedy 2opt in CUDA to solve mQAP

Program to solve instances of multiobjective quadratic assignment problems in CUDA C

What it includes:

1. Creation of initial population.
2. Fitness calculation for each objective.
3. Parllel NSGA-II:
  - [3.1] Dominance matrix.
  - [3.2] Total dominance.
  - [3.3] Pareto fronts.
  - [3.4] Rank.
  - [3.5] Pupulation fitness ordered by objectives (will be used in crowding calculation).
  - [3.6] Crowding distance.
  - [3.7] Get offspring population.
 4. Binary tournament selection.
 5. Mutation.
 6. Addapted Greedy 2opt. 