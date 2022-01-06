#ifndef __DP_HPP__
#define __DP_HPP__

#include <list>
#include <utility>
#include <vector>
#include <iostream>
#include <iterator>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "score.hpp"
#include "LTSS.hpp"

#define UNUSED(expr) do { (void)(expr); } while (0)

using namespace Objectives;

class DPSolver {
public:
  DPSolver(std::vector<float> a,
	   std::vector<float> b,
	   int T,
	   objective_fn parametric_dist=objective_fn::Gaussian,
	   bool risk_partitioning_objective=false,
	   bool use_rational_optimization=false
	   ) :
    n_{static_cast<int>(a.size())},
    T_{T},
    a_{a},
    b_{b},
    optimal_score_{0.},
    parametric_dist_{parametric_dist},
    risk_partitioning_objective_{risk_partitioning_objective},
    use_rational_optimization_{use_rational_optimization}
    
  { _init(); }

  DPSolver(int n,
	   int T,
	   std::vector<float> a,
	   std::vector<float> b,
	   objective_fn parametric_dist=objective_fn::Gaussian,
	   bool risk_partitioning_objective=false,
	   bool use_rational_optimization=false
	   ) :
    n_{n},
    T_{T},
    a_{a},
    b_{b},
    optimal_score_{0.},
    parametric_dist_{parametric_dist},
    risk_partitioning_objective_{risk_partitioning_objective},
    use_rational_optimization_{use_rational_optimization}
    
  { _init(); }

  std::vector<std::vector<int> > get_optimal_subsets_extern() const;
  float get_optimal_score_extern() const;
  std::vector<float> get_score_by_subset_extern() const;
  void print_maxScore_();
  void print_nextStart_();
    
private:
  int n_;
  int T_;
  std::vector<float> a_;
  std::vector<float> b_;
  std::vector<std::vector<float> > maxScore_, maxScore_sec_;
  std::vector<std::vector<int> > nextStart_, nextStart_sec_;
  std::vector<int> priority_sortind_;
  float optimal_score_;
  std::vector<std::vector<int> > subsets_;
  std::vector<float> score_by_subset_;
  objective_fn parametric_dist_;
  bool risk_partitioning_objective_;
  bool use_rational_optimization_;
  std::unique_ptr<ParametricContext> context_;
  std::unique_ptr<LTSSSolver> LTSSSolver_;

  void _init() { 
    create();
    optimize();
  }
  void create();
  void createContext();
  void create_multiple_clustering_case();
  void optimize();
  void optimize_multiple_clustering_case();

  void sort_by_priority(std::vector<float>&, std::vector<float>&);
  void reorder_subsets(std::vector<std::vector<int> >&, std::vector<float>&);
  float compute_score(int, int);
  float compute_ambient_score(float, float);
};


#endif
