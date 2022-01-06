#ifndef __PYTHON_DPSOLVER_HPP__
#define __PYTHON_DPSOLVER_HPP__

#include "score.hpp"
#include "DP.hpp"
#include "threadpool.hpp"
#include "threadsafequeue.hpp"

#include <vector>
#include <utility>
#include <limits>
#include <type_traits>

std::vector<std::vector<int> > find_optimal_partition__DP(int n,
							 int T,
							 std::vector<float> a,
							 std::vector<float> b,
							 int parametric_dist,
							 bool risk_partitioning_objective,
							 bool use_rational_optimization
							 );

float find_optimal_score__DP(int n,
			     int T,
			     std::vector<float> a,
			     std::vector<float> b,
			     int parametric_dist,
			     bool risk_partitioning_objective,
			     bool use_rational_optimization);

std::pair<std::vector<std::vector<int> >, float> optimize_one__DP(int n,
								 int T,
								 std::vector<float> a,
								 std::vector<float> b,
								 int parametric_dist,
								 bool risk_partitioning_objective,
								 bool use_rational_optimization);

std::pair<std::vector<std::vector<int>>, float> sweep_best__DP(int n,
							       int T,
							       std::vector<float> a,
							       std::vector<float> b,
							       int parametric_dist,
							       bool risk_partitioning_objective,
							       bool use_rational_optimization);

std::vector<std::pair<std::vector<std::vector<int>>, float>> sweep_parallel__DP(int n,
										int T,
										std::vector<float> a,
										std::vector<float> b,
										int parametric_dist,
										bool risk_partitioning_objective,
										bool use_rational_optimization);


#endif
