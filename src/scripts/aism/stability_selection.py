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
from sklearn.svm import SVC

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
    auc = metrics.roc_auc_score(y, y_pred)
    return {'accuracy': acc, 'precision': prec,
            'recall': rcll, 'MCC': mcc, 'AUC': auc}


def stability_selection(estimator, X, y):
    """Naive stability selection process with MC samples."""
    rs = StratifiedShuffleSplit(n_splits=2, test_size=.25)
    feats = []
    test_metrics = []
    print('      + Running {} splits:'.format(rs.n_splits))
    for i, (train_index, test_index) in enumerate(rs.split(X, y)):
        tic = time.time()
        estimator.fit(X.iloc[train_index], y.iloc[train_index])
        # Save the features
        _mdl = estimator.best_estimator_.steps[1][1]
        if hasattr(_mdl, 'coef_'):
            feats.append(X.columns[np.nonzero(_mdl.coef_)])
        elif hasattr(_mdl, 'support_'):
            feats.append(X.columns[_mdl.support_])

        test_metrics.append(evaluate(estimator, X.iloc[test_index], y.iloc[test_index]))
        print('      + Split {}/{} done [{}].'.format(i+1, rs.n_splits, str(datetime.timedelta(seconds=time.time()-tic))))
    return feats, test_metrics


def create_pipelines():
    """Create all the models organized in a dictionary."""
    names = [
        'gradient_boosting',
        'random_forests',
        'l1l2',
        'l2_logistic_regression',
        'l1_logistic_regression',
        'linear_svc'
        ]

    estimators = [
        RFECV(GradientBoostingClassifier(learning_rate=0.05), step=.25, cv=3, n_jobs=-1),
        RFECV(RandomForestClassifier(), step=.25, cv=3, n_jobs=-1),
        L1L2Classifier(),
        LogisticRegression(penalty='l2'),
        LogisticRegression(penalty='l1'),
        RFECV(SVC(kernel='linear'), step=.25, cv=3, n_jobs=-1)
        ]

    params = [
        {'predict__estimator__max_depth': map(int, np.linspace(50, 100, 10)),  # gradient_boosting
         'predict__estimator__n_estimators': map(int, np.linspace(10, 200, 10))},
        {'predict__estimator__max_features': map(int, np.linspace(0.1, 0.8, 10)),  # random_forests
         'predict__estimator__min_samples_leaf': map(int, np.linspace(5, 100, 10)),
         'predict__estimator__n_estimators': [1000]},
        {'predict__alpha': np.logspace(-3, 2, 30),  # l1l2
         'predict__l1_ratio': np.linspace(1e-3, 1, 30)},
        {'predict__alpha': np.logspace(-3, 2, 30)},  # l2 logistic regression
        {'predict__alpha': np.logspace(-3, 2, 30)},  # l1 logistic regression
        {'predict__estimator__C': np.logspace(-3, 3, 30)}  # linear svr
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


def main():
    """Stability selection main routine."""
    print('-----------------------------------------')
    print('Stability selection on the AISM dataset')
    print('-----------------------------------------\n')

    # 0. Load data
    data = pd.read_csv('dataset_12-2017/data_training.csv', header=0, index_col=0)
    labels = pd.read_csv('dataset_12-2017/labels_training.csv', header=0, index_col=0)
    print('- Data loaded: {} x {}'.format(*data.shape))

    # 1. Define the cross-validated pipelines
    pipes = create_pipelines()
    print('- {} pipelines created'.format(len(pipes)))

    # 2. Iterate over the pipelines
    print('- Starting main routine:')
    for key in pipes.keys():
        print('    * Running {}...'.format(key))
        feats, scores = stability_selection(pipes[key], data, labels)
        print('    * done.')
        print('    * Dumping results on disk...')
        dump(feats, key+'_coefs.pkl')
        dump(scores, key+'_scores.pkl')
        print('    * done.')
    print('done.')

################################################################################

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
