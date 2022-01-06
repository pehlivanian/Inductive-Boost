#ifndef __PYTHON_LTSSSOLVER_HPP__
#define __PYTHON_LTSSSOLVER_HPP__

#include "LTSS.hpp"

#include <vector>
#include <utility>
#include <type_traits>

std::vector<int> find_optimal_partition__LTSS(int n,
					      std::vector<float> a,
					      std::vector<float> b,
					      int parametric_dist);

float find_optimal_score__LTSS(int n,
			       std::vector<float> a,
			       std::vector<float> b,
			       int parametric_dist);

std::pair<std::vector<int>, float> optimize_one__LTSS(int n,
						      std::vector<float> a,
						      std::vector<float> b,
						      int parametric_dist);

#endif
