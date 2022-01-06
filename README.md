# Inductive Boosting
Engine and driver for inductive boosting. Each iterative step of gradient boosting, XGBoost, LightGBM, Catboost, etc. requires minimization of a quadratic approximation to the loss function. We propose an exact solution to the optimization program, then inductively fit a classifier to the specified partition, in some cases achieiving superior results to classic gradient boosting (represented by catboost in the included test suite).

The make target is mainly the SWIG bindings for the python interface, as displayed in solver_ex.py, pmlb_driver.py. There is a full C++ interface that's built along with it, but it'sonly used in the google tests right now.

### Requirements:
- cmake
- swig
- theano
- catboost
- seaborn

### Optional:
- google mock, test
- Stree
_ pmlb

### Compile to nonlocal ./build with C++17-compliant compiler, tests: 
```
$ cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release -DGTEST=ON -DUSE_C++17=ON
$ cmake --build build -- -j4
```

### C++ Unittests:
```
$ ./build/bin/gtest_all
```

#### Python unittests:
```
$ python solver_ex.py
```

Examples of python calling conventions are contained in pmlb_driver.py which performs successive fits over pmlb datasets.

