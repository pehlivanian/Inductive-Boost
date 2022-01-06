import heapq
import numpy as np
from functools import partial
from itertools import islice
import multiprocessing
import sklearn.base
import sklearn.tree
import theano
import theano.tensor as T
import logging

import classifier
import solverSWIG_DP

SEED = 515
rng = np.random.RandomState(SEED)

class Distribution:
    GAUSSIAN = 0
    POISSON = 1
    RATIONALSCORE = 2

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.WARN)

def gen_blocks(num_items, block_size):
    bin_ends = list(range(0,num_items, int(num_items/block_size)))
    bin_ends = bin_ends + [num_items] if num_items/block_size else bin_ends
    islice_on = list(zip(bin_ends[:-1], bin_ends[1:]))
    
    slices = [list(islice(range(num_items), *ind)) for ind in islice_on]
    return slices

class end_task(object):
    pass

class OptimalSplitGradientBoostingClassifier(object):
    def __init__(self,
                 X,
                 y=None,
                 min_partition_size=10,
                 max_partition_size=25,
                 row_sample_ratio=1.,
                 col_sample_ratio=1.,
                 gamma=0.1,
                 eta=0.1,
                 num_classifiers=100,
                 use_constant_term=False,
                 solver_type='linear_hessian',
                 learning_rate=0.1,
                 distiller=classifier.classifierFactory(sklearn.tree.DecisionTreeClassifier),
                 use_closed_form_differentials=False,
                 risk_partitioning_objective=False,
                 ):
        ############
        ## Inputs ##
        ############
        self.X = theano.shared(value=X, name='X', borrow=True)
        self.X_all = theano.shared(value=X, name='X_all', borrow=True)
        self.y = theano.shared(value=y, name='y', borrow=True)
        self.y_all = theano.shared(value=y, name='y_all', borrow=True)
        initial_X = self.X.get_value()
        self.N, self.num_features = initial_X.shape
        self.N_all, self.num_features_all = self.N, self.num_features
        
        self.min_partition_size = min_partition_size
        self.max_partition_size = max_partition_size
        self.row_sample_ratio = row_sample_ratio
        self.col_sample_ratio = col_sample_ratio
        self.gamma = gamma
        self.eta = eta
        self.num_classifiers = num_classifiers
        self.curr_classifier = 0

        # solver_type is one of
        # ('quadratic, 'linear_hessian', 'linear_constant')
        self.use_constant_term = use_constant_term
        self.solver_type = solver_type
        self.learning_rate = learning_rate
        self.distiller = distiller
        self.use_closed_form_differentials = use_closed_form_differentials
        self.risk_partitioning_objective = risk_partitioning_objective
        ################
        ## END Inputs ##
        ################

        # optimal partition at each step, not part of any gradient, not a tensor
        self.partitions = list()
        # distinct leaf values at each step, not a tesnor
        self.distinct_leaf_values = np.zeros((self.num_classifiers + 1,
                                             self.N))
        # regularization penalty at each step
        self.regularization = theano.shared(name='regularization',
                                            value=np.zeros((self.num_classifiers + 1,
                                                            1)).astype(theano.config.floatX))
        # leaf values at each step
        self.leaf_values = theano.shared(name='leaf_values',
                                         value=np.zeros((self.num_classifiers + 1,
                                                         self.N,
                                                         1)).astype(theano.config.floatX))
        # classifier at each step
        self.implied_trees = [classifier.LeafOnlyTree(self.X.get_value(), 0.5 * np.ones((self.N, 1)))] * \
                             (self.num_classifiers + 1)

        # Set initial set of leaf_values to be random
        # XXX
        # Was self.min_partition_size
        if isinstance(self.distiller.args[0], sklearn.base.ClassifierMixin):
            # Cannot have number of unique classes == number of samples, so
            # we must restrict sampling to create fewer classes
            leaf_value = rng.choice(rng.uniform(low=0.0, high=1.0, size=(self.min_partition_size,)),
            # leaf_value = rng.choice(rng.uniform(low=0.0, high=1.0, size=(self.max_partition_size,)),                                    
                                    size=(self.N, 1)).astype(theano.config.floatX)
        elif isinstance(self.distiller.args[0], sklearn.base.RegressorMixin):
            leaf_value = np.asarray(rng.uniform(low=0.0, high=1.0, size=(self.N, 1))).astype(theano.config.floatX)
        else:
            raise RuntimeError('Cannot determine whether distiller is of classifier or regressor type.')
        self.set_next_leaf_value(leaf_value)

        # Set initial partition to be the size 1 partition (all leaf values the same)
        self.partitions.append(list(range(self.N)))

        # Set initial classifier
        implied_tree = self.imply_tree(leaf_value)
        
        self.set_next_classifier(implied_tree)        
        
        # For testing
        self.srng = T.shared_randomstreams.RandomStreams(seed=SEED)

        # Set initial predictions
        self.curr_predictions = np.zeros((self.num_classifiers+1,
                                          self.N,
                                          1)).astype(theano.config.floatX)
        self.curr_predictions[0] = theano.function([], implied_tree.predict(self.X))()

        # masks
        self.row_mask = list(range(self.N))
        self.col_mask = list(range(self.num_features))

        self.curr_classifier += 1

    def set_next_classifier(self, classifier):
        i = self.curr_classifier
        self.implied_trees[i] = classifier

    def set_next_leaf_value(self, leaf_value):
        i = self.curr_classifier
        c = T.dmatrix()
        update = (self.leaf_values,
                  T.set_subtensor(self.leaf_values[i, :, :], c))
        f = theano.function([c], updates=[update])
        f(leaf_value)

    def imply_tree(self, leaf_values, **impliedSolverKwargs):
        X0 = self.X.get_value()
        y0 = leaf_values
        return self.distiller(X0, y0, **impliedSolverKwargs)
            
    def weak_learner_predict(self, classifier_ind):
        classifier = self.implied_trees[classifier_ind]
        return classifier.predict(self.X)

    class predict_task(object):
        def __init__(self, i, X0, classifier):
            self.i = i
            self.X0 = X0
            self.classifier = classifier

        def __call__(self):
            return self._task(self.X0)

        def _task(self, X0):
            return self.classifier.predict(X0, numpy_format=True)

    class predict_worker(multiprocessing.Process):
        def __init__(self, task_queue, result_queue):
            multiprocessing.Process.__init__(self)
            self.task_queue = task_queue
            self.result_queue = result_queue

        def run(self):
            proc_name = self.name
            while True:
                task = self.task_queue.get()
                if isinstance(task, end_task):
                    self.task_queue.task_done()
                    break
                result = task()
                self.task_queue.task_done()
                self.result_queue.put(result)

    def predict_from_input(self, X0):
        NUM_BLOCKS = min(20, self.curr_classifier)
        CONCURRENCY = 10

        y_hat = theano.shared(name='y_hat', value=np.zeros((X0.shape[0], 1)))
        slices = gen_blocks(self.curr_classifier, NUM_BLOCKS)

        for slice_ind, slice in enumerate(slices):
            tasks = multiprocessing.JoinableQueue()
            results = multiprocessing.Queue()
            workers = [self.predict_worker(tasks, results) for _ in range(min([CONCURRENCY, self.curr_classifier]))]
        
            for worker in workers:
                worker.start()
            
            for ind in slice:
                implied_tree = self.implied_trees[ind]
                tasks.put(self.predict_task(ind, X0, implied_tree))

            for _ in workers:
                tasks.put(end_task())
            
            tasks.join()

            allResults = list()
            while not results.empty():
                result = results.get(block=True)
                y_hat += theano.shared(value=result.astype(theano.config.floatX))
                # logging.warn('QUEUE SIZE: {}'.format(results.qsize()))
            logging.warn('FINISHED WITH SLICE: {}/{}'.format(slice_ind, -1+len(slices)))
            del tasks
            del results
            del workers

        return y_hat
        
    def predict_from_input_serialized(self, X0):
        X = theano.shared(value=X0.astype(theano.config.floatX))
        y_hat = theano.shared(name='y_hat', value=np.zeros((X0.shape[0], 1)))
        for classifier_ind in range(self.curr_classifier):
            y_hat += self.implied_trees[classifier_ind].predict(X, numpy_format=False)
        return y_hat
        
    def predict_old(self):
        y_hat = theano.shared(name='y_hat', value=np.zeros((self.N, 1)))
        for classifier_ind in range(self.curr_classifier):
            y_hat += self.implied_trees[classifier_ind].predict(self.X)
        return y_hat

    def predict(self):
        # col_mask filtering handled differently; masked columns
        # are 0-filled to supress
        # XXX
        # Above method doesn't work with tree classifier, e.g.
        y_hat = np.zeros((self.N, 1))
        for classifier_ind in range(self.curr_classifier):
            y_hat += self.curr_predictions[classifier_ind][self.row_mask,:]
        return theano.shared(name='y_hat', value=y_hat, borrow=True)

    def predict_scan(self):
        def iter_step(classifier_ind):
            y_step = self.weak_learner_predict(classifier_ind)
            # y_step = self.implied_trees[classifier_ind].predict(self.X)
            return y_step

        # scan is short-circuited by length of T.arange(self.curr_classifier)
        y,inner_updates = theano.scan(
            fn=iter_step,
            sequences=[T.arange(self.curr_classifier)],
            outputs_info=[None],
            )

        return T.sum(y, axis=0)

    def fit(self, num_steps=None):
        num_steps = num_steps or self.num_classifiers

        # Initial leaf_values for initial loss calculation
        leaf_values = self.leaf_values.get_value()[0,:]
        print('STEP {}: LOSS: {:4.6f}'.format(0,
                                              theano.function([],
                                                              self.loss(
                                                                  self.predict(),
                                                                  len(np.unique(leaf_values)),
                                                                  leaf_values))()))
        # Iterative boosting
        for i in range(1,num_steps):
            self.fit_step()
            leaf_values = self.leaf_values.get_value()[-1+self.curr_classifier,:]
            print('STEP {}: LOSS: {:4.6f}'.format(i,
                                                  theano.function([],
                                                                  self.loss(
                                                                      self.predict(),
                                                                      len(np.unique(leaf_values)),
                                                                      leaf_values))()))
            # Summary statistics mid-training
        print('Training finished')
    def find_best_optimal_split_old(self, g, h, num_partitions):
        ''' Method: results contains the optimal partitions for all partition sizes in
            [1, num_partitions]. We take each, from the optimal_split_tree from an
            inductive fitting of the classifier, then look at the loss of the new
            predictor (current predictor + optimal_split_tree predictor). The minimal
            loss wins.
        '''

        logging.warn('NUM_PARTITIONS: {}'.format(num_partitions))
        sweep_mode = True
        results = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                              g,
                                              h,
                                              objective_fn=Distribution.RATIONALSCORE,
                                              risk_partitioning_objective=self.risk_partitioning_objective,
                                              use_rational_optimization=True,
                                              sweep_mode=sweep_mode)()

        logging.info('found optimal partition')

        npart = T.scalar('npart')
        lv = T.dmatrix('lv')
        x = T.dmatrix('x')
        loss = theano.function([x,npart,lv], self.loss(x, npart, lv))

        loss_heap = []

        results = results

        for rind, result in enumerate(results):
            leaf_values = np.zeros((self.N, 1))
            subsets = result[0]
            for subset in subsets:
                s = list(subset)

                min_val = -1 * np.sum(g[s])/(np.sum(h[s]) + self.gamma)
                
                # XXX
                # impliedSolverKwargs = dict(max_depth=max([int(len(s)/2), 2]))
                # impliedSolverKwargs = dict(max_depth=int(np.log2(num_partitions)))
                impliedSolverKwargs = dict(max_depth=None)
                leaf_values[s] = self.learning_rate * min_val
            optimal_split_tree = self.imply_tree(leaf_values, **impliedSolverKwargs)
            loss_new = loss(theano.function([], self.predict())() +
                            theano.function([], optimal_split_tree.predict(self.X))(),
                            len(subsets),
                            leaf_values)
            heapq.heappush(loss_heap, (loss_new.item(0), rind, leaf_values))

        best_loss, best_rind, best_leaf_values = heapq.heappop(loss_heap)

        logging.warn('BEST RIND: {}'.format(best_rind))

        logging.info('found optimal leaf values')

        # XXX
        # solverKwargs = dict(max_depth=int(np.log2(num_partitions)))
        solverKwargs = dict(max_depth=None)        
        optimal_split_tree = self.imply_tree(best_leaf_values, **solverKwargs)

        # ===============================
        # == If DecisionTreeClassifier ==
        # ===============================
        # from sklearn import tree
        # import graphviz
        # dot_data = tree.export_graphviz(tr, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render('Boosting')

        self.partitions.append(results[best_rind][0])

        # XXX
        # implied_values = theano.function([], optimal_split_tree.predict(self.X))()
        # logging.info('found implied values for comparison')
        # print('leaf_values:    {!r}'.format([(round(val,4), np.sum(best_leaf_values==val))
        #                                     for val in np.unique(best_leaf_values)]))
        # print('implied_values: {!r}'.format([(round(val,4), np.sum(implied_values==val))
        #                                     for val in np.unique(implied_values)]))

        return best_leaf_values


    def find_best_optimal_split(self, g, h, num_partitions):
        ''' Method: results contains the optimal partitions for all partition sizes in
            [1, num_partitions]. We take each, from the optimal_split_tree from an
            inductive fitting of the classifier, then look at the loss of the new
            predictor (current predictor + optimal_split_tree predictor). The minimal
            loss wins.
        '''

        logging.warn('NUM_PARTITIONS: {}'.format(num_partitions))

        sweep_mode = False
        results = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                              g,
                                              h,
                                              objective_fn=Distribution.RATIONALSCORE,
                                              risk_partitioning_objective=self.risk_partitioning_objective,
                                              use_rational_optimization=True,
                                              sweep_mode=sweep_mode)()

        logging.info('found optimal partition')
        assert(len(results[0]) == num_partitions)               

        npart = T.scalar('npart')
        lv = T.dmatrix('lv')
        x = T.dmatrix('x')
        loss = theano.function([x,npart,lv], self.loss(x, npart, lv))
        yhat = theano.function([], self.predict())()
        yhat0 = np.asarray(yhat)

        loss_heap = []

        results = (results,)

        for rind, result in enumerate(results):
            leaf_values = np.zeros((self.N, 1))
            subsets = result[0]
            for subset in subsets:
                s = list(subset)

                min_val = -1 * np.sum(g[s])/(np.sum(h[s]) + self.gamma)
                MAX_VAL = 0.9
                MIN_VAL = 0.1
                
                # XXX
                # impliedSolverKwargs = dict(max_depth=max([int(len(s)/2), 2]))
                # impliedSolverKwargs = dict(max_depth=int(np.log2(num_partitions)))
                impliedSolverKwargs = dict(max_depth=None)

                # bounded
                # pre_leaf_values = np.zeros((self.N,1))
                # pre_leaf_values[s] = np.minimum(MAX_VAL-yhat0[s], np.array([self.learning_rate*min_val]*len(s)).reshape(-1,1))
                # pre_leaf_values[s] = np.maximum(MIN_VAL-yhat0[s], np.array([self.learning_rate*min_val]*len(s)).reshape(-1,1))
                # leaf_values[s] = pre_leaf_values[s]

                # unbounded
                leaf_values[s] = self.learning_rate * min_val
            optimal_split_tree = self.imply_tree(leaf_values, **impliedSolverKwargs)
            loss_new = loss(yhat +
                            theano.function([], optimal_split_tree.predict(self.X))(),
                            len(subsets),
                            leaf_values)
            heapq.heappush(loss_heap, (loss_new.item(0), rind, leaf_values))

        best_loss, best_rind, best_leaf_values = heapq.heappop(loss_heap)

        logging.info('found optimal leaf values')

        # XXX
        # solverKwargs = dict(max_depth=int(np.log2(num_partitions)))
        solverKwargs = dict(max_depth=None)        
        optimal_split_tree = self.imply_tree(best_leaf_values, **solverKwargs)

        # ===============================
        # == If DecisionTreeClassifier ==
        # ===============================
        # from sklearn import tree
        # import graphviz
        # dot_data = tree.export_graphviz(tr, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render('Boosting')

        self.partitions.append(results[best_rind][0])

        # XXX
        # implied_values = theano.function([], optimal_split_tree.predict(self.X))()
        # logging.info('found implied values for comparison')
        # print('leaf_values:    {!r}'.format([(round(val,4), np.sum(best_leaf_values==val))
        #                                     for val in np.unique(best_leaf_values)]))
        # print('implied_values: {!r}'.format([(round(val,4), np.sum(implied_values==val))
        #                                     for val in np.unique(implied_values)]))

        return best_leaf_values

    from contextlib import contextmanager

    @contextmanager
    def _subsample_rows(self):

        mask = sorted(rng.choice(self.N, size=int(self.N * self.row_sample_ratio), replace=False))

        self.X = theano.shared(value=self.X_all.get_value()[mask,:], name='X', borrow=True)
        self.y = theano.shared(value=self.y_all.get_value()[mask], name='y', borrow=True)
        self.N = int(self.N * self.row_sample_ratio)

        try:
            yield mask
        finally:
            pass

    @contextmanager
    def _subsample_columns(self):
        ''' Must be called after _subsample_rows, if that is called. We handle subsampling
            of columns differently than subsampling of rows as the number of features is
            built in to the classifier.
        '''

        mask = sorted(rng.choice(self.num_features,
                                 size=int(self.num_features * (1. - self.col_sample_ratio)), replace=False))

        X0 = self.X.get_value()
        # X0[:,sorted(set(range(self.num_features)) - set(mask))] = 0
        X0[:, sorted(set(mask))] = 0
        self.X = theano.shared(value=X0, name='X', borrow=True)

        try:
            yield mask
        finally:
            pass
        
    def fit_step(self):

        self.X_all = self.X
        self.y_all = self.y
        self.N_all = self.N

        self.row_mask,self.col_mask = None,None
        
        with self._subsample_rows() as self.row_mask:
            logging.info('set row_mask')
            
            with self._subsample_columns() as self.col_mask:
                logging.info('set col_mask')
                g, h, c = self.generate_coefficients(constantTerm=self.use_constant_term,
                                                     row_mask=self.row_mask)
                logging.info('generated coefficients')
            
                # SWIG optimizer, task-based C++ distribution
                num_partitions = max(1, int(rng.choice(range(max(self.min_partition_size, self.max_partition_size)))))
        
                # Find best optimal split
                best_leaf_values = self.find_best_optimal_split(g, h, num_partitions)
                
        self.X = self.X_all
        self.y = self.y_all
        self.N = self.N_all
        
        best_leaf_values_all = np.zeros((self.N, 1))
        if self.row_mask:
            best_leaf_values_all[self.row_mask,:] = best_leaf_values
        best_leaf_values = best_leaf_values_all
        
        # Set leaf_value, return leaf values used to generate
        self.set_next_leaf_value(best_leaf_values)

        # Calculate optimal_split_tree
        # XXX
        # impliedSolverKwargs = dict(max_depth=int(np.log2(num_partitions)))
        impliedSolverKwargs = dict(max_depth=None)        
        optimal_split_tree = self.imply_tree(best_leaf_values, **impliedSolverKwargs)

        # Set implied_tree
        self.set_next_classifier(optimal_split_tree)

        # Set current marginal prediction
        self.curr_predictions[self.curr_classifier] = theano.function([],
                                                                 optimal_split_tree.predict(self.X))()

        self.row_mask = list(range(self.N))
        self.col_mask = list(range(self.num_features))

        self.curr_classifier += 1

    def generate_coefficients(self, row_mask=None, constantTerm=False):

        if (self.use_closed_form_differentials):
            if len(self.col_mask) == 0:
                # If no column mask, use cached predictions and restrict by row
                # depending on row mask
                # g_f = self.grad_exp_loss_without_regularization(self.predict())
                # h_f = self.hess_exp_loss_without_regularization(self.predict())
                # g_f = self.grad_logit_loss_without_regularization(self.predict())
                # h_f = self.hess_logit_loss_without_regularization(self.predict())
                g_f = self.grad_mse_loss_without_regularization(self.predict())
                h_f = self.hess_mse_loss_without_regularization(self.predict())
                # g_f = self.grad_cosh_loss_without_regularization(self.predict())
                # h_f = self.hess_cosh_loss_without_regularization(self.predict())
            else:
                # If column mask, cannot rely on cached predictions
                # g_f = self.grad_exp_loss_without_regularization(self.predict_from_input(self.X.get_value()))
                # h_f = self.hess_exp_loss_without_regularization(self.predict_from_input(self.X.get_value()))
                # g_f = self.grad_logit_loss_without_regularization(self.predict_from_input(self.X.get_value()))
                # h_f = self.hess_logit_loss_without_regularization(self.predict_from_input(self.X.get_value()))
                g_f = self.grad_mse_loss_without_regularization(self.predict_from_input(self.X.get_value()))
                h_f = self.hess_mse_loss_without_regularization(self.predict_from_input(self.X.get_value()))
                # g_f = self.grad_cosh_loss_without_regularization(self.predict_from_input(self.X.get_value()))
                # h_f = self.hess_cosh_loss_without_regularization(self.predict_from_input(self.X.get_value()))

            g = theano.function([], g_f)()[0]
            h = theano.function([], h_f)()[0]
            c = None
            if constantTerm and not self.solver_type == 'linear_hessian':
                c = theano.function([], self._mse_coordinatewise(self.predict()))().squeeze()

            return (g, h, c)        

        else:

            def fn(x):
                return x*(x-1)*(x-2)

            # Attempt at global approximation
            if False:
                xaxis = np.arange(-1., 1.1, .1)
                leaf_values = self.leaf_values.get_value()[-1+self.curr_classifier,:]
                y_hat0 = theano.function([], self.predict())().reshape(-1)
                leaf_values = self.leaf_values.get_value()[-1+self.curr_classifier,:]
                
                pf = list()
                for ind in range(y_hat0.shape[0]):                
                    yaxis = [np.asscalar(theano.function([], self.loss(
                        np.concatenate([y_hat0[:ind], xaxis[xind:(xind+1)], y_hat0[ind+1:]]),
                        len(np.unique(leaf_values)),
                        leaf_values
                        ))()) for xind in range(0,len(xaxis))]
                    # cp = np.polynomial.Chebyshev.fit(xaxis, yaxis, 2)
                    pf.append(np.polynomial.polynomial.polyfit(xaxis,yaxis,2))
                    print('ind: {}'.format(ind))
                    import pdb; pdb.set_trace()
                
            x = T.dvector('x')
            leaf_values = self.leaf_values.get_value()[-1+self.curr_classifier,:]
            if row_mask:
                leaf_values = leaf_values[row_mask,:]
            loss = self.loss(T.shape_padaxis(x, 1), len(np.unique(leaf_values)), leaf_values)

            grads = T.grad(loss, x)
            hess = T.hessian(loss, x)
            
            G = theano.function([x], grads)
            H = theano.function([x], hess)
            y_hat0 = theano.function([], self.predict())().reshape(-1)
            g = G(y_hat0)
            h = np.diag(H(y_hat0))
            h = h + np.array([self.gamma]*h.shape[0])

            c = None
            if constantTerm and not self.solver_type == 'linear_hessian':
                c = theano.function([], self._mse_coordinatewise(self.predict()))().squeeze()

            return (g, h, c)
        
    def loss(self, y_hat, num_partitions, leaf_values):
        return self.loss_without_regularization(y_hat) + self.regularization_loss(num_partitions, leaf_values)

    def loss_without_regularization(self, y_hat):
        ''' Dependent on loss function '''
        return self.mse_loss_without_regularization(y_hat)
        # XXX
        # return self.exp_loss_without_regularization(y_hat)
        # return self.logit_loss_without_regularization(y_hat)
        # return self.cross_entropy_loss_without_regularization(y_hat)
        # return self.cosh_loss_without_regularization(y_hat)

    def regularization_loss(self, num_partitions, leaf_values):
        ''' Independent of loss function '''
        size_reg = self.gamma * num_partitions
        coeff_reg = 0.5 * self.eta * T.sum(T.extra_ops.Unique(False,False,False)(leaf_values)**2)
        return size_reg + coeff_reg

    def mse_loss_without_regularization(self, y_hat):
        return self._mse(y_hat)

    def logit_loss_without_regularization(self, y_hat):
        return T.sum(T.log(1+T.exp(-(2*y_hat.T-1)*(2*self.y-1)).T))

    def grad_mse_loss_without_regularization(self, y_hat):
        return -2*(self.y - y_hat.T)

    def hess_mse_loss_without_regularization(self, y_hat):
        # return theano.shared(name='h', value=2.*np.ones((1,self.N)))
        return 2.*T.ones((1,self.N)).astype('float64')
    
    def grad_logit_loss_without_regularization(self, y_hat):
        f1 = T.exp(-(2*y_hat.T-1)*(2*self.y-1))
        # num = -2*(2*self.y-1)*f1
        # den = 1 + f1
        # return num/den
        return (-2*(2*self.y-1)*f1)/(1 + f1)        

    def hess_logit_loss_without_regularization(self, y_hat):
        f1 = T.exp(-(2*y_hat.T-1)*(2*self.y-1))
        f15 = (1+f1)
        f2 = (2*self.y-1)*(2*self.y-1)
        f3 = (4*f2*f1)
        f4 = f3/f15
        return f4-f4/f15
        # num = (1+f1)*(4*f2*f1)-(4*f2*f1*f1)
        # den = (1+f1)*(1+f1)
        # return num/den            
    
    def exp_loss_without_regularization(self, y_hat):
        return T.sum(T.exp(-1(2*y_hat.T-1)*(2*self.y-1)))

    def grad_exp_loss_without_regularization(self, y_hat):
        return -2*(2*self.y-1)*T.exp(-(2*y_hat.T-1)*(2*self.y-1))

    def hess_exp_loss_without_regularization(self, y_hat):
        return 4*(2*self.y-1)*(2*self.y-1)*T.exp(-(2*y_hat.T-1)*(2*self.y-1))
    
    def exp_loss_without_regularization(self, y_hat):
        # return T.sum(T.exp(-y_hat * T.shape_padaxis(self.y, 1)))
        return T.sum(T.exp(-(2*y_hat.T-1)*(2*self.y-1)).T)

    def cosh_loss_without_regularization_coordinatewise(self, y, y_hat):
        return T.log(T.cosh(y_hat-y))

    def cosh_loss_without_regularization(self, y_hat):
        return T.sum(T.log(T.cosh(y_hat - T.shape_padaxis(self.y, 1))))

    def grad_cosh_loss_without_regularization(self, y_hat):
        return T.sinh(y_hat.T - self.y)/T.cosh(y_hat.T - self.y)

    def hess_cosh_loss_without_regularization(self, y_hat):
        return 2/(1 + T.cosh(2*(y_hat.T - self.y)))
    
    def hinge_loss_without_regularization(self, y_hat):
        return T.sum(T.abs_(y_hat - T.shape_padaxis(self.y, 1)))

    def cross_entropy_loss_without_regularization(self, y_hat):
        y0 = T.shape_padaxis(self.y, 1)
        # return T.sum(-(y0 * T.log(y_hat) + (1-y0)*T.log(1-y_hat)))
        return T.sum(T.nnet.binary_crossentropy(y_hat, y0))
    
    def _mse(self, y_hat):
        # return T.sqrt(T.sum(self._mse_coordinatewise(y_hat)))
        return T.sum(self._mse_coordinatewise(y_hat))

    def _mse_coordinatewise(self, y_hat):
        return (T.shape_padaxis(self.y, 1) - y_hat)**2

    def quadratic_solution_scalar(self, g, h, c):
        a,b = 0.5*h, g
        s1 = -b
        s2 = np.sqrt(b**2 - 4*a*c)
        r1 = (s1 + s2) / (2*a)
        r2 = (s1 - s2) / (2*a)

        return r1, r2
    
    def quadratic_solution( self, g, h, c):
        a,b = 0.5*h, g
        s1 = -b
        s2 = np.sqrt(b**2 - 4*a*c)
        r1 = (s1 + s2) / (2*a)
        r2 = (s1 - s2) / (2*a)

        return (r1.reshape(-1,1), r2.reshape(-1, 1))
