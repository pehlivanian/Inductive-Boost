import numpy as np
import solverSWIG_DP
import proto

rng = np.random.RandomState(136)

num_partitions = 10
n = 500
objective_fn = 1                    # 1 ~ Poisson, 2 ~ Gaussian
risk_partitioning_objective = False # False => multiple clustering score function is used
optimized_score_calculation = False # Leave this False; only implemented for RationalScore case

a_lower_limit = 0. if objective_fn == 1 else -10.; a_higher_limit = 10.
b_lower_limit = 0.; b_higher_limit = 10.
a = rng.uniform(low=a_lower_limit, high=a_higher_limit, size=n)
b = rng.uniform(low=b_lower_limit, high=b_higher_limit, size=n)


# all_results[0] ~ size n partition
# all_results[1] ~ cumulative score
all_results = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                          a,
                                          b,
                                          objective_fn,
                                          risk_partitioning_objective,
                                          optimized_score_calculation)()
best_result_sweep = solverSWIG_DP.OptimizerSWIG(len(a)-10,
                                          a,
                                          b,
                                          objective_fn,
                                          risk_partitioning_objective,
                                          optimized_score_calculation,
                                          True)()


print("OPTIMAL PARTITION")
print("=================")
print('{!r}'.format(all_results[0]))
print('SCORE: {}'.format(all_results[1]))
print("BEST RESULT SWEEP")
print("=================")
print('{!r}'.format(best_result_sweep[0]))
print('SCORE: {}'.format(best_result_sweep[1]))
print('PARTITION SIZE: {}'.format(len(best_result_sweep[0])))
