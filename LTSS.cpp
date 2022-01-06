#include <algorithm>
#include <iterator>
#include <numeric>
#include <limits>
#include <iostream>
#include <utility>
#include <cmath>

#include "LTSS.hpp"


struct distributionException : public std::exception {
  const char* what() const throw () {
    return "Bad distributional assignment";
  };
};


float
LTSSSolver::compute_score(int i, int j) {
  return context_->compute_score(i, j);
}

void
LTSSSolver::sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);

  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });

  priority_sortind_ = ind;

  // Inefficient reordering
  std::vector<float> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
  
}

void
LTSSSolver::createContext() {
  // create reference to score function
  if (parametric_dist_ == objective_fn::Gaussian) {
    context_ = std::make_unique<GaussianContext>(a_, 
						 b_, 
						 n_, 
						 false,
						 false);
  }
  else if (parametric_dist_ == objective_fn::Poisson) {
    context_ = std::make_unique<PoissonContext>(a_, 
						b_, 
						n_,
						false,
						false);
  }
  else if (parametric_dist_ == objective_fn::RationalScore) {
    context_ = std::make_unique<RationalScoreContext>(a_,
						      b_,
						      n_,
						      false,
						      false);
  }
  else {
    throw distributionException();
  }
}

void 
LTSSSolver::create() {
  // sort by priority
  sort_by_priority(a_, b_);

  subset_ = std::vector<int>();

  // create context
  createContext();
}

void
LTSSSolver::optimize() {
  optimal_score_ = 0.;

  float maxScore = -std::numeric_limits<float>::max();
  std::pair<int, int> p;
  // Test ascending partitions
  for (int i=1; i<=n_; ++i) {
    auto score = compute_score(0, i);
    if (score > maxScore) {
      maxScore = score;
      p = std::make_pair(0, i);
    }
  }
  // Test descending partitions
  for (int i=n_-1; i>=0; --i) {
    auto score = compute_score(i, n_);
    if (score > maxScore) {
      maxScore = score;
      p = std::make_pair(i, n_);
    }
  }
  
  for (int i=p.first; i<p.second; ++i) {
    subset_.push_back(priority_sortind_[i]);
  }
  optimal_score_ = maxScore;
}

std::vector<int>
LTSSSolver::get_optimal_subset_extern() const {
  return subset_;
}

float
LTSSSolver::get_optimal_score_extern() const {
  return optimal_score_;
}
