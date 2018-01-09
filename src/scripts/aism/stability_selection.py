#!/usr/bin/env python
"""Perform selection stability on a bunch of models on the AISM dataset."""
from __future__ import print_function
import warnings
import time
import datetime

import numpy as np
import pandas as pd

from sklearn import metrics
from l1l2py.classification import L1L2Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Re-use the utilities of the dataprep script
from aism_dataprep import dump


def evaluate(estimator, X, y):
    """Compute the selected metrics on the input (test) set."""
    y_pred = estimator.predict(X)
    # metrics
    acc = metrics.accuracy_score(y, y_pred)
    prec = metrics.precision_score(y, y_pred)
    rcll = metrics.recall_score(y, y_pred)
    mcc = metrics.matthews_corrcoef(y, y_pred)
    auc = metrics.roc_auc_score(y, y_pred) # FIXME this index is nonsensical
    return {'accuracy': acc, 'precision': prec,
            'recall': rcll, 'MCC': mcc, 'AUC': auc}


def stability_selection(estimator, X, y, tag=None):
    """Naive stability selection process with MC samples."""
    rs = StratifiedShuffleSplit(n_splits=100, test_size=.25)
    feats = []
    scores = []
    print('      + Running {} splits:'.format(rs.n_splits))
    for i, (train_index, test_index) in enumerate(rs.split(X, y)):
        tic = time.time()
        estimator.fit(X.iloc[train_index], y.iloc[train_index])
        # Save the features
        _mdl = estimator.best_estimator_.steps[1][1]
        if hasattr(_mdl, 'coef_'):
            assert(len(_mdl.coef_.ravel()) == 165)  # sanity check
            feats.append(X.columns[np.nonzero(_mdl.coef_.ravel())[0]])
        elif hasattr(_mdl, 'support_'):
            assert(len(_mdl.support_.ravel()) == 165)  # sanity check
            feats.append(X.columns[_mdl.support_])

        scores.append(evaluate(estimator, X.iloc[test_index], y.iloc[test_index]))
        print('      + [{}] Split {}/{} done [{}].'.format(tag, i+1, rs.n_splits, str(datetime.timedelta(seconds=time.time()-tic))))

    # Save result
    if tag is not None:
        print('    * [{}] Dumping results on disk...'.format(tag))
        dump(feats, tag+'_coefs_MM.pkl')
        dump(scores, tag+'_scores_MM.pkl')
        print('    * [{}] done.'.format(tag))
    #return feats, scores


def create_pipelines():
    """Create all the models organized in a dictionary."""
    names = [
        'gradient_boosting',
        'random_forests',
        'l1l2',
        'l2_logistic_regression',
        'l1_logistic_regression',
        'linear_svc_l2',
        'linear_svc_l1',
        ]

    estimators = [
        RFECV(GradientBoostingClassifier(learning_rate=0.05,  # enable early stopping for gradient boosting
                                         n_iter_no_change=10,  # this needs sklearn > 0.20dev
                                         validation_fraction=0.2,
                                         n_estimators=500), step=.25, cv=3, n_jobs=-1),
        RFECV(RandomForestClassifier(), step=.25, cv=3, n_jobs=-1),
        L1L2Classifier(),
        RFECV(LogisticRegression(penalty='l2'), step=.25, cv=3, n_jobs=-1),
        LogisticRegression(penalty='l1'),
        RFECV(LinearSVC(penalty='l2'), step=.25, cv=3, n_jobs=-1),
        LinearSVC(penalty='l1', dual=False)
        ]

    params = [
        {'predict__estimator__max_depth': map(int, np.linspace(10, 100, 15)),  # gradient_boosting
         # 'predict__estimator__n_estimators': map(int, np.linspace(10, 500, 15)),
         'predict__estimator__max_features': ['sqrt', 0.5, None]},
        {'predict__estimator__max_features': np.linspace(0.1, 0.9, 10),  # random_forests
         'predict__estimator__min_samples_leaf': np.arange(1, 10),
         'predict__estimator__n_estimators': [500]},
        {'predict__alpha': np.logspace(-3, 2, 30),  # l1l2
         'predict__l1_ratio': np.linspace(1e-3, 1, 30)},
        {'predict__estimator__C': np.logspace(-3, 2, 30)},  # l2 logistic regression
        {'predict__C': np.logspace(-3, 2, 30)},  # l1 logistic regression
        {'predict__estimator__C': np.logspace(-3, 3, 15)},  # linear svc l2
        {'predict__C': np.logspace(-3, 3, 15)}  # linear svc l1
        ]

    # Create all the cross-validated pipeline objects
    pipes = {}
    for name, estimator, param in zip(names, estimators, params):
        pipe = Pipeline([['preproc', StandardScaler()],
                         ['predict', estimator]])
        pipes[name] = GridSearchCV(estimator=pipe,
                                   param_grid=param,
                                   n_jobs=-1)
    return pipes


def master_parallel(pipes, X, y):
    """Parallel pipelines evaluation."""
    import multiprocessing as mp
    jobs = []

    # Submit jobs
    for i, key in enumerate(pipes.keys()):
        proc = mp.Process(target=stability_selection,
                          args=(pipes[key], X, y, key))
        jobs.append(proc)
        proc.start()
        print("- Job: %s submitted", i)

    # Collect results
    count = 0
    for proc in jobs:
        proc.join()
        count += 1
    print("- %d jobs collected", count)

    # import joblib as jl
    # jl.Parallel(n_jobs=-1) \
    #     (jl.delayed(pipe_worker)(
    #         'pipe' + str(i), pipe, pipes_dump, X) for i, pipe in enumerate(
    #             pipes))



def main():
    """Stability selection main routine."""
    print('-----------------------------------------')
    print('Stability selection on the AISM dataset')
    print('-----------------------------------------\n')

    # 0. Load data
    data = pd.read_csv('dataset_12-2017/data_training.csv', header=0, index_col=0)
    labels = pd.read_csv('dataset_12-2017/labels_training.csv', header=0, index_col=0)
    yy = np.where(labels.values == 'SP', 1, 0)  # map RR - SP / 0 - 1
    labels = pd.DataFrame(data=yy, index=labels.index, columns=labels.columns)
    print('- Data loaded: {} x {}'.format(*data.shape))

    # 1. Define the cross-validated pipelines
    pipes = create_pipelines()
    print('- {} pipelines created'.format(len(pipes)))

    # 2. Iterate over the pipelines
    print('- Starting main routine:')
    master_parallel(pipes, data, labels)
    # for key in pipes.keys():
    #     print('    * Running {}...'.format(key))
    #     feats, scores = stability_selection(pipes[key], data, labels)
    #     print('    * done.')
    #     print('    * Dumping results on disk...')
    #     dump(feats, key+'_coefs.pkl')
    #     dump(scores, key+'_scores.pkl')
    #     print('    * done.')
    print('done.')

################################################################################

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
