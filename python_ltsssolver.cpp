#include "python_ltsssolver.hpp"

std::vector<int> find_optimal_partition__LTSS(int n,
					      std::vector<float> a,
					      std::vector<float> b,
					      int parametric_dist) {

  auto ltss = LTSSSolver(n, a, b, static_cast<objective_fn>(parametric_dist));
  return ltss.get_optimal_subset_extern();

}

float find_optimal_score__LTSS(int n,
			       std::vector<float> a,
			       std::vector<float> b,
			       int parametric_dist) {
  auto ltss = LTSSSolver(n, a, b, static_cast<objective_fn>(parametric_dist));
  return ltss.get_optimal_score_extern();
}

std::pair<std::vector<int>, float> optimize_one__LTSS(int n,
						      std::vector<float> a,
						      std::vector<float> b,
						      int parametric_dist) {
  auto ltss = LTSSSolver(n, a, b, static_cast<objective_fn>(parametric_dist));
  std::vector<int> subset = ltss.get_optimal_subset_extern();
  float score = ltss.get_optimal_score_extern();
  return std::make_pair(subset, score);
}
