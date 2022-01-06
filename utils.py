from catboost import CatBoostClassifier, Pool
from itertools import combinations
from scipy.special import comb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
import theano
import theano.tensor as T


#########################
## Partition Utilities ##
#########################
def monotonic_partitions(n, m):
    ''' Returns endpoints of all monotonic
        partitions
    '''
    combs = combinations(range(n-1), m-1)
    parts = list()
    for comb in combs:
        yield [(l+1, r+1) for l,r in zip((-1,)+comb, comb+(n-1,))]

def knuth_partitions(ns, m):
    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)

def Bell_n_k(n, k):
    ''' Number of partitions of {1,...,n} into
        k subsets, a restricted Bell number
    '''
    if (n == 0 or k == 0 or k > n): 
        return 0
    if (k == 1 or k == n): 
        return 1

    return (k * Bell_n_k(n - 1, k) + 
                Bell_n_k(n - 1, k - 1))

def Mon_n_k(n, k):
    return comb(n-1, k-1, exact=True)

#############################
## END Partition Utilities ##
#############################

##############
## Graphics ##
##############

def plot_confusion(confusion_matrix, source_class_names, target_class_names, figsize=(10,7), fontsize=14, title='Confusion Matrix'):
    df_cm = pd.DataFrame(
        confusion_matrix, index=source_class_names, columns=target_class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    return fig

##################
## END Graphics ##
##################

###############
## Summaries ##
###############

def oos_summary(clf, X_train, y_train, X_test, y_test, learning_rate=1.,
                catboost_iterations=100,
                catboost_depth=None,
                catboost_learning_rate=0.5,
                catboost_loss_function='CrossEntropy',                
                catboost_verbose=False
                ):
    # Vanilla regression model
    X0 = clf.X.get_value()
    y0 = clf.y.get_value()
    reg = LinearRegression(fit_intercept=True).fit(X0, y0)
    logreg = LogisticRegression().fit(X0, y0)
    clf_cb = CatBoostClassifier(iterations=100,
                                depth=None,
                                learning_rate=learning_rate,
                                loss_function=catboost_loss_function,
                                verbose=False)
    
    clf_cb.fit(X0, y0)
    
    x = T.dmatrix('x')
    _loss = theano.function([x], clf.loss_without_regularization(x))
    
    y_hat_clf = np.asarray(theano.function([], clf.predict())())
    y_hat_ols = reg.predict(X0).reshape(-1,1)
    y_hat_lr = logreg.predict(X0).reshape(-1,1)
    y_hat_cb = clf_cb.predict(X0).reshape(-1,1)

    y_hat_clf = (y_hat_clf > .5).astype(int)
    y_hat_ols = (y_hat_ols > .5).astype(int)

    igb_loss_IS = _loss(y_hat_clf)
    ols_loss_IS = _loss(y_hat_ols)
    lr_loss_IS = _loss(y_hat_lr)
    cb_loss_IS = _loss(y_hat_cb)

    # igb_acc_IS = metrics.accuracy_score(y_hat_clf, y_train)
    # cp_acc_IS = metrics.accuracy_sscore(y_hat_cb, y_train)

    print('IS _loss_clf: {:4.6f}'.format(_loss(y_hat_clf)))
    print('IS _loss_ols: {:4.6f}'.format(_loss(y_hat_ols)))
    print('IS _loss_lr:  {:4.6f}'.format(_loss(y_hat_lr)))
    print('IS _loss_cb:  {:4.6f}'.format(_loss(y_hat_cb)))
    print()

    # Out-of-sample predictions
    X0 = X_test
    y0 = y_test.reshape(-1,1)
    
    y_hat_clf = theano.function([], clf.predict_from_input(X0))()
    y_hat_ols = reg.predict(X0).reshape(-1,1)
    y_hat_lr = logreg.predict(X0).reshape(-1,1)
    y_hat_cb = clf_cb.predict(X0).reshape(-1,1)

    y_hat_clf = (y_hat_clf > .5).astype(int)
    y_hat_ols = (y_hat_ols > .5).astype(int)
                        
    def _loss(y_hat):
        return np.sum((y0 - y_hat)**2)
    
    print('OOS _loss_clf: {:4.6f}'.format(_loss(y_hat_clf)))
    print('OOS _loss_ols: {:4.6f}'.format(_loss(y_hat_ols)))
    print('OOS _loss_lr:  {:4.6f}'.format(_loss(y_hat_lr)))
    print('OOS _loss_cb:  {:4.6f}'.format(_loss(y_hat_cb)))
    print()
    
    print('OOS _accuracy_clf: {:1.4f}'.format(metrics.accuracy_score(y_hat_clf, y0)))
    print('OOS _accuracy_ols: {:1.4f}'.format(metrics.accuracy_score(y_hat_ols, y0)))
    print('OOS _accuracy_lr:  {:1.4f}'.format(metrics.accuracy_score(y_hat_lr, y0)))
    print('OOS _accuracy_cb:  {:1.4f}'.format(metrics.accuracy_score(y_hat_cb, y0)))

    return (igb_loss_IS,
            cb_loss_IS,
            metrics.accuracy_score(y_hat_clf, y0),
            metrics.accuracy_score(y_hat_cb, y0))
    
    # target_names = ['0', '1']
    # conf = plot_confusion(metrics.confusion_matrix(y_hat_clf, y0), target_names)
    # plt.show()

###################
## END Summaries ##
###################
