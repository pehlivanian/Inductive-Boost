#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <limits>
#include <Python.h>

#include "score.hpp"
#include "DP.hpp"

// RAII for embedded Python
class PyContext {
public:
  PyContext() { Py_Initialize(); }
  ~PyContext() { Py_Finalize(); }
};

void sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);
  
  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });
  std::vector<float> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
	    
}

void pretty_print_subsets(std::vector<std::vector<int> >& subsets) {
  std::cout << "SUBSETS\n";
  std::cout << "[\n";
  std::for_each(subsets.begin(), subsets.end(), [](std::vector<int>& subset){
		  std::cout << "[";
		  std::copy(subset.begin(), subset.end(),
			    std::ostream_iterator<int>(std::cout, " "));
		  std::cout << "]\n";
		});
  std::cout << "]" << std::endl;
}

float rational_obj(std::vector<float> a, std::vector<float> b, int start, int end) {
  if (start == end)
    return 0.;
  float den = 0., num = 0.;
  for (int ind=start; ind<end; ++ind) {
    num += a[ind];
    den += b[ind];
  }
  return num*num/den;
}

TEST(DPSolverTest, OptimizationFlag) {

  int n = 100, T = 25;
  size_t NUM_CASES = 100;
  
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.), distb(0., 10.);

  std::vector<float> a(n), b(n);

  for (size_t i=0; i<NUM_CASES; ++i) {
    for (auto &el : a)
      el = dista(gen);
    for (auto&el : b)
      el = distb(gen);

    std::vector<objective_fn> objectives = {objective_fn::Gaussian,
					    objective_fn::Poisson,
					    objective_fn::RationalScore};

    for (auto objective : objectives) {
    
      auto dp_unopt = DPSolver(n, T, a, b, objective, true, false);
      auto dp_opt = DPSolver(n, T, a, b, objective, true, true);
      
      auto subsets_unopt = dp_unopt.get_optimal_subsets_extern();
      auto subsets_opt = dp_opt.get_optimal_subsets_extern();
    
      ASSERT_EQ(subsets_unopt.size(), subsets_opt.size());
      
      for (size_t j=0; j<subsets_unopt.size(); ++j) {
	ASSERT_EQ(subsets_unopt[j], subsets_opt[j]);
      }
      
      dp_unopt = DPSolver(n, T, a, b, objective, false, false);
      dp_opt = DPSolver(n, T, a, b, objective, false, true);
      
      subsets_unopt = dp_unopt.get_optimal_subsets_extern();
      subsets_opt = dp_opt.get_optimal_subsets_extern();
      
      ASSERT_EQ(subsets_unopt.size(), subsets_opt.size());
      
      for (size_t j=0; j<subsets_unopt.size(); ++j) {
	ASSERT_EQ(subsets_unopt[j], subsets_opt[j]);
      }    
    }
  }
}

TEST(DPSolverTest, Baselines ) {

  std::vector<float> a{0.0212651 , -0.20654906, -0.20654906, -0.20654906, -0.20654906,
      0.0212651 , -0.20654906,  0.0212651 , -0.20654906,  0.0212651 ,
      -0.20654906,  0.0212651 , -0.20654906, -0.06581402,  0.0212651 ,
      0.03953075, -0.20654906,  0.16200014,  0.0212651 , -0.20654906,
      0.20296943, -0.18828341, -0.20654906, -0.20654906, -0.06581402,
      -0.20654906,  0.16200014,  0.03953075, -0.20654906, -0.20654906,
      0.03953075,  0.20296943, -0.20654906,  0.0212651 ,  0.20296943,
      -0.20654906,  0.0212651 ,  0.03953075, -0.20654906,  0.03953075};
  std::vector<float> b{0.22771114, 0.21809504, 0.21809504, 0.21809504, 0.21809504,
      0.22771114, 0.21809504, 0.22771114, 0.21809504, 0.22771114,
      0.21809504, 0.22771114, 0.21809504, 0.22682739, 0.22771114,
      0.22745816, 0.21809504, 0.2218354 , 0.22771114, 0.21809504,
      0.218429  , 0.219738  , 0.21809504, 0.21809504, 0.22682739,
      0.21809504, 0.2218354 , 0.22745816, 0.21809504, 0.21809504,
      0.22745816, 0.218429  , 0.21809504, 0.22771114, 0.218429  ,
      0.21809504, 0.22771114, 0.22745816, 0.21809504, 0.22745816};

  std::vector<std::vector<int> > expected = {
    {1, 2, 3, 4, 6, 8, 10, 12, 16, 19, 22, 23, 25, 28, 29, 32, 35, 38, 21},
    {13, 24}, 
    {0, 5, 7, 9, 11, 14, 18, 33, 36, 15, 27, 30, 37, 39},
    {17, 26}, 
    {20, 31, 34}
  };

  std::vector<float> a1{2.26851454, 2.86139335, 5.51314769, 6.84829739, 6.96469186, 7.1946897,
      9.80764198, 4.2310646};
  std::vector<float> b1{3.43178016, 3.92117518, 7.29049707, 7.37995406, 4.80931901, 4.38572245,
      3.98044255, 0.59677897};

  auto dp1 = DPSolver(8, 3, a1, b1, objective_fn::Poisson, false, false);
  auto opt1 = dp1.get_optimal_subsets_extern();
  
  // sort_by_priority(a, b);

  auto dp = DPSolver(40, 5, a, b, objective_fn::Gaussian, true, false);
  auto opt = dp.get_optimal_subsets_extern();

  for (size_t i=0; i<expected.size(); ++i) {
    auto expected_subset = expected[i], opt_subset = opt[i];
    ASSERT_EQ(expected_subset.size(), opt_subset.size());
    for(size_t j=0; j<expected_subset.size(); ++j) {
      ASSERT_EQ(expected_subset[j], opt_subset[j]);
    }
  }
}

TEST(DPSolverTest, OrderedProperty) {
  // Case (n,T) = (50,5)
  int n = 100, T = 20;
  
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(1., 10.);

  std::vector<float> a(n), b(n);

  for (size_t i=0; i<5; ++i) {
    for (auto &el : a)
      el = dist(gen);
    for (auto &el : b)
      el = dist(gen);

    // Presort
    sort_by_priority(a, b);

    auto dp = DPSolver(n, T, a, b, objective_fn::Gaussian, false, false);
    auto opt = dp.get_optimal_subsets_extern();

    int sum;
    std::vector<int> v;

    for (auto& list : opt) {
      if (list.size() > 1) {
	v.resize(list.size());
	std::adjacent_difference(list.begin(), list.end(), v.begin());
	sum = std::accumulate(v.begin()+1, v.end(), 0);
      }
    }

    // We ignored the first element as adjacent_difference has unintuitive
    // result for first element
    ASSERT_EQ(sum, v.size()-1);
  }
  
}

TEST(DPSolverTest, OptimalityTestWithRandomPartitions) {
  int NUM_CASES = 1000, NUM_SUBCASES = 500, T = 3;

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_int_distribution<int> distn(5, 50);
  std::uniform_real_distribution<float> dista( 1., 10.);
  std::uniform_real_distribution<float> distb( 1., 10.);

  std::vector<float> a, b;

  for (int case_num=0; case_num<NUM_CASES; ++case_num) {

    int n = distn(gen);
    a.resize(n); b.resize(n);

    for (auto &el : a)
      el = dista(gen);
    for (auto &el : b)
      el = distb(gen);
  
    sort_by_priority(a, b);

    ASSERT_GE(n, 5);
    ASSERT_LE(n, 100);

    auto dp = DPSolver(n, T, a, b, objective_fn::Poisson, true, false);
    auto dp_opt = dp.get_optimal_subsets_extern();
    auto scores = dp.get_score_by_subset_extern();

    for (int subcase_num=0; subcase_num<NUM_SUBCASES; ++subcase_num) {
      std::uniform_int_distribution<int> distm(5, n);
      
      int m1 = distm(gen), m21;
      int m2 = distm(gen), m22;
      if ((m1 == m2) || (m1 == n) || (m2 == n))
	continue;
      m21 = std::min(m1, m2);
      m22 = std::max(m1, m2);
      
     
      auto context = PoissonContext(a,
				    b,
				    n,
				    true,
				    false);

      float rand_score, dp_score;
      rand_score = context.compute_score_riskpart(0, m21) + 
	context.compute_score_riskpart(m21, m22) + 
	context.compute_score_riskpart(m22, n);
      dp_score = context.compute_score_riskpart(dp_opt[0][0], 1+dp_opt[0][dp_opt[0].size()-1]) + 
	context.compute_score_riskpart(dp_opt[1][0], 1+dp_opt[1][dp_opt[1].size()-1]) + 
	context.compute_score_riskpart(dp_opt[2][0], 1+dp_opt[2][dp_opt[2].size()-1]);

      if ((dp_score - rand_score) > std::numeric_limits<float>::epsilon()) {
	ASSERT_LE(rand_score, dp_score);
      }
    }
  }
}

auto main(int argc, char **argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
