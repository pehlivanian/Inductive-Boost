from functools import partial
import importlib
import numpy as np
from catboost import CatBoostClassifier, Pool
from stree import Stree
import sklearn.base
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import if_delegate_has_method
import theano
import theano.tensor as T

class InductiveBase(BaseEstimator):
    def __init__(self, classifier, X, y, **solverKwargs):
        self.X = X
        self.y = y
        self.classifier_ = clone(classifier)
        for attr_,val_ in solverKwargs.items():
            setattr(self.classifier_, attr_, val_)

        assert not any(np.isnan(self.y))
            
    def fit(self, **fitKwargs):
        self.classifier_.fit(self.X, self.y, **fitKwargs)

class InductiveRegressor(InductiveBase):
    ''' Decorator class to turn transductive regressor
        into an inductive one.
    '''
    def __init__(self, classifier, X, y, **solverKwargs):
        super(InductiveRegressor, self).__init__(classifier, X, y, **solverKwargs)
        self.fit()

    @if_delegate_has_method(delegate='classifier_')
    def predict(self, X):
        yhat0 = self.classifier_.predict(X.get_value())
        yhat = T.as_tensor(yhat0.reshape(-1,1).astype(theano.config.floatX))
        return yhat

class InductiveClassifier(InductiveBase):
    ''' Decorator class to turn transductive classifier
        into an inductive one.
    '''
    
    def __init__(self, classifier, X, y, **solverKwargs):
        super(InductiveClassifier, self).__init__(classifier, X, y, **solverKwargs)
        
        unique_vals = np.unique(y)
        self.val_to_class = dict(zip(unique_vals, range(len(unique_vals))))
        self.class_to_val = {v:k for k,v in self.val_to_class.items()}
        self.y = np.array([self.val_to_class[x[0]] for x in y])
        
        assert not any(np.isnan(self.y))
        self.fit()

    @if_delegate_has_method(delegate='classifier_')
    def predict(self, X, numpy_format=False):
        if numpy_format:
            yhat0 = self.classifier_.predict(X)
            return np.array([self.class_to_val[x] for x in yhat0]).reshape(-1, 1).astype(theano.config.floatX)
        yhat0 = self.classifier_.predict(X.get_value())        
        yhat = T.as_tensor(np.array([self.class_to_val[x]
                                     for x in yhat0]).reshape(-1, 1).astype(theano.config.floatX))            
        return yhat


class CatBoostImpliedTree(CatBoostClassifier):
    def __init__(self, X=None, y=None, max_depth=2):
        super(CatBoostImpliedTree, self).__init__(iterations=100,
                                                  depth=2,
                                                  learning_rate=1,
                                                  loss_function='MultiClass',
                                                  verbose=False)

        self.fit(X, y)

    def predict(self, X):
        y_hat0 = super(CatBoostImpliedTree, self).predict(X.get_value())
        y_hat = T.as_tensor(y_hat0.reshape(-1,1)).astype(theano.config.floatX)
        return y_hat

class STreeImpliedTree(Stree):
    def __init__(self, X=None, y=None, max_depth=2):
        super(STreeImpliedTree, self).__init__(random_state=random_state,
                                               max_features='auto')
        self.fit(X, y)

    def predict(self, X):
        y_hat0 = super(STreeImpliedTree, self).predict(X.get_value())
        y_hat = T.as_tensor(y_hat0.reshape(-1,1)).astype(theano.config.floatX)
        return y_hat

class LeafOnlyTree(object):
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y


    def predict(self, X):
        return self.y
    
def classifierFactory(clz, **modelArgs):
    if isinstance(clz(), sklearn.base.RegressorMixin):
        return partial(InductiveRegressor, clz(**modelArgs))
    elif isinstance(clz(), sklearn.base.ClassifierMixin):
        return partial(InductiveClassifier, clz(**modelArgs))
    else:
        raise RuntimeError('Cannot determine whether {} is a classifier or a regressor'.format(clz))

# linear_model = importlib.import_module('sklearn.linear_model')
# discriminant_analysis = importlib.import_module('sklearn.discriminant_analysis')
# support_vector_machine = importlib.import_module('sklearn.svm')
# tree = importlib.import_module('sklearn.tree')
    
# OLS Linear Regression
# LinearRegressorKwargs =  {'fit_intercept': True }
# LinearRegressor = partial(InductiveRegressor, linear_model.LinearRegression())

# LDA
# LinearDiscriminantKwargs = {}
# LinearDiscriminant = partial(InductiveClassifier, discriminant_analysis.LinearDiscriminantAnalysis())

# Support Vector Machine
# SVCKwargs = {}
# SVC = partial(InductiveClassifier, support_vector_machine.SVC())

# Decision Tree
# DecisionTreeKwargs = {}
# DecisionTree = partial(InductiveClassifier, tree.DecisionTreeClassifier())

