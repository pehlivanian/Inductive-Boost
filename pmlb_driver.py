import pmlb

import logging
import classifier
import numpy as np
import pandas as pd
import sklearn.tree
import sklearn.svm
import sklearn.discriminant_analysis
from stree import Stree
from sklearn.model_selection import train_test_split
import utils
from optimalsplitboost import OptimalSplitGradientBoostingClassifier

TEST_SIZE = 0.20
FILENAME = './summary.csv'
USE_SIMULATED_DATA = False

########################
## PMLB Dataset sizes ##
########################

# To determine which datasets are associated with 0-1 classification problems
if False:
    class_datasets = set()
    from pmlb import classification_dataset_names, regression_dataset_names
    for dataset_name in classification_dataset_names:
        X,y = pmlb.fetch_data(dataset_name, return_X_y=True)
        if len(np.unique(y)) == 2:
            class_datasets.add(dataset_name)
else:
    class_datasets = {'analcatdata_lawsuit', 'analcatdata_boxing2', 'heart_c', 'agaricus_lepiota', 'horse_colic', 'australian', 'flare', 'phoneme', 'breast_cancer_wisconsin', 'analcatdata_creditscore', 'clean1', 'monk2', 'analcatdata_aids', 'analcatdata_bankruptcy', 'churn', 'hypothyroid', 'lupus', 'GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1', 'labor', 'analcatdata_boxing1', 'xd6', 'Hill_Valley_without_noise', 'corral', 'adult', 'molecular_biology_promoters', 'twonorm', 'breast_w', 'bupa', 'diabetes', 'chess', 'german', 'mux6', 'backache', 'tokyo1', 'ionosphere', 'prnn_crabs', 'monk3', 'mofn_3_7_10', 'threeOf9', 'ring', 'spect', 'biomed', 'colic', 'sonar', 'hepatitis', 'cleve', 'monk1', 'irish', 'parity5+5', 'coil2000', 'magic', 'breast_cancer', 'analcatdata_cyyoung8092', 'GAMETES_Epistasis_3_Way_20atts_0.2H_EDM_1_1', 'GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1', 'postoperative_patient_data', 'crx', 'clean2', 'house_votes_84', 'dis', 'haberman', 'saheart', 'GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1', 'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001', 'profb', 'analcatdata_fraud', 'tic_tac_toe', 'kr_vs_kp', 'credit_g', 'prnn_synth', 'credit_a', 'parity5', 'mushroom', 'breast', 'glass2', 'pima', 'analcatdata_asbestos', 'appendicitis', 'vote', 'analcatdata_japansolvent', 'heart_h', 'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM_2_001', 'wdbc', 'buggyCrx', 'analcatdata_cyyoung9302', 'hungarian', 'spambase', 'spectf', 'heart_statlog', 'Hill_Valley_with_noise'}

class_datasets = sorted(class_datasets, key=lambda x: x.upper())

already_read = set([])

# Exceptions
carve_out = set(('adult','agaricus_lepiota', 'churn', 'clean1', 'clean2', 'coil2000', 'magic', 'postoperative_patient_data', 'ring'))

df = pd.DataFrame(columns=['dataset', 'row_ratio', 'col_ratio', 'part_ratio', 'learning_rate', 'numsteps', 'rat', 'features', 'classes', 'imbalance', 'igb_loss_IS', 'cbt_loss_IS', 'igb_acc_OS', 'cbt_acc_OS'])

for ind,dataset_name in enumerate(class_datasets):

    if dataset_name in carve_out.union(already_read):
        continue

    print('PROCESSING DATASET: {}'.format(dataset_name))

    if (USE_SIMULATED_DATA): 
        from sklearn.datasets import make_classification, load_breast_cancer
        from sklearn.model_selection import train_test_split
        SEED = 0xFACE0FF
        NUM_SAMPLES = 10000
        NUM_FEATURES = 20
        rng = np.random.RandomState(SEED)
        
        X,y = make_classification(random_state=SEED, n_samples=NUM_SAMPLES, n_features=NUM_FEATURES)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    else:
        X,y = pmlb.fetch_data(dataset_name, return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)    

    # XXX
    distiller = classifier.classifierFactory(sklearn.tree.DecisionTreeClassifier)
    # distiller = classifier.classifierFactory(Stree)
    num_steps = 250 # 250
    part_ratio = .1 # .5
    min_partition_size= 2
    max_partition_size = int(part_ratio*X_train.shape[0])
    row_sample_ratio = 0.5 # 0.65
    col_sample_ratio = 1.0
    gamma = 0.01
    eta = 0.025
    learning_rate = 0.10 # 0.15
    distiller = distiller
    use_closed_form_differentials = True

    clfKwargs = { 'min_partition_size':            min_partition_size,
                  'max_partition_size':            max_partition_size,
                  'row_sample_ratio':              row_sample_ratio,
                  'col_sample_ratio':              col_sample_ratio,
                  'gamma':                         gamma,
                  'eta':                           eta,
                  'num_classifiers':               num_steps,
                  'use_constant_term':             False,
                  'solver_type':                   'linear_hessian',
                  'learning_rate':                 learning_rate,
                  'distiller':                     distiller,
                  'use_closed_form_differentials': True, # Only available for some loss functions
                  'risk_partitioning_objective':   False,
                  }

    clf = OptimalSplitGradientBoostingClassifier( X_train, y_train, **clfKwargs)

    clf.fit(num_steps)

    stats = utils.oos_summary(clf, X_train, y_train, X_test, y_test,
                              catboost_iterations=num_steps,
                              catboost_depth=None,
                              catboost_learning_rate=learning_rate,
                              catboost_loss_function='Logloss',
                              catboost_verbose=False)

    all_stats = [dataset_name, row_sample_ratio, col_sample_ratio, part_ratio, learning_rate, num_steps,
                 X_train.shape[0], X_train.shape[1], len(np.unique(y_train)),
                 2.0*((sum(y_train==0)/len(y_train) - .5)**2 + (sum(y_train==1)/len(y_train) - .5)**2),
                 stats[0], stats[1], stats[2], stats[3]]

    df.loc[ind] = all_stats
    df.to_csv(FILENAME)
    logging.warning('Finished: %s', dataset_name)
    
