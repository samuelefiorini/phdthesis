#!/usr/bin/env python
"""Challenge different ML methods on the aging problem."""

from __future__ import print_function

import os
import warnings

import distributed.joblib # maybe useless
from joblib import Parallel # maybe useless
from joblib import parallel_backend # the only one actually relevant

import cPickle as pkl
import numpy as np
import pandas as pd

from fancyimpute import KNN
from hurry.filesize import size
from l1l2py.classification import L1L2Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def flatten(x):
    """Flatten a list."""
    return [y for l in x for y in flatten(l)] if type(x) in (list, np.ndarray) else [x]


def load_raw_data():
    """Load the raw data of the aism problem via pandas."""
    return pd.read_csv(os.path.join('..', '..' , 'data', 'AISM',
                                    'anonimyzed_data_12-2017.csv'),
                       header=0, index_col=0)


def load_labels():
    """Load the labels of the aism problem via pandas."""
    return pd.read_csv(os.path.join('..', '..' , 'data', 'AISM',
                                    'anonimyzed_labels_12-2017.csv'),
                       header=0, index_col=0)


def training_validation_test_split(data):
    """Return the index of training/validation/test data.

    Exams 1,2,3    -> training
    Exam  4        -> validation
    Exam  5,...,11 -> test
    """
    idx_train = filter(lambda x: x.endswith('S01') or x.endswith('S02') or x.endswith('S031'),
                       data.index)
    idx_valid = filter(lambda x: x.endswith('S04'),
                       data.index)
    idx_test = filter(lambda x: x not in flatten([idx_train, idx_valid]),
                      data.index)
    return idx_train, idx_valid, idx_test


def preprocess(data):
    """Categorical feature mapping + KNN imputing.

    dfx: input data DataFrame object with shape (n,d)
    """
    # 1. Impute missing values via KNN
    data = pd.DataFrame(data=KNN(k=3).complete(data.values),
                        index=data.index, columns=data.columns)

    # 2. One-Hot-Encode of EDINB
    ohe = OneHotEncoder(sparse=False, n_values=3)
    edi_items = filter(lambda x: x.startswith('EDINB0'), data.columns)
    ohe_edi_items = flatten([[x+'_0', x+'_1', x+'_2'] for x in edi_items])

    xx = data[edi_items].values
    xx[np.where(xx == -2.0)] = 1.0  # get rid of pesky negative values
    xx = ohe.fit_transform(xx)

    return pd.concat([data.drop(edi_items, axis=1),
                      pd.DataFrame(xx, columns=ohe_edi_items, index=data.index)
                      ], axis=1)


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
        GradientBoostingClassifier(learning_rate=0.05),
        RFECV(RandomForestClassifier(), step=.25, cv=3),
        L1L2Classifier(),
        LogisticRegression(penalty='l2'),
        LogisticRegression(penalty='l1'),
        RFECV(SVC(kernel='linear'), step=.25, cv=3)
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


def dump(object_to_dump, filename):
    """Dump the input object on disk."""
    with open(filename, 'wb') as f:
        pkl.dump(object_to_dump, f)
    print('- {} dumped now {}'.format(filename, size(os.path.getsize(filename))))


def main():
    """Model challenge main routine."""
    print('--------------------------------------')
    print('The AISM problem model challenge')
    print('--------------------------------------\n')

    # 0. Load data
    data = load_raw_data()
    labels = load_labels()
    print('- Raw data loaded: {} samples / {} features.'.format(*data.shape))

    # 1. Preprocess data
    data = preprocess(data)
    print('- Data preprocessed: {} samples / {} features.'.format(*data.shape))

    # 2. Train/validation/test split
    idx_train, _, _ = training_validation_test_split(data)
    print(idx_test)
    return

    # 2. Create pipelines
    pipes = create_pipelines()
    print('- {} pipelines created'.format(len(pipes)))
    return

    # 3. Perform the model challenge twice
    cross_val_scores = ('accuracy', 'precision', 'recall', 'mcc')

    # with parallel_backend('dask.distributed', scheduler_host='megazord:8786'):
    # Individually evaluate the performance of each pipeline
    scores_dump = {}
    for pipe in pipes.keys():
        print('Fitting {}...'.format(pipe))
        scores = cross_validate(estimator=pipes[pipe],
                                X=data.values, y=labels.values.ravel(),
                                scoring=cross_val_scores,
                                cv=ShuffleSplit(n_splits=500, test_size=.25),
                                n_jobs=-1, verbose=1)

        scores_dump[pipe] = scores # save results
        dump(scores_dump, 'scores.pkl')
        print('Done in {:3.5} sec.\n'.format(np.sum(scores['fit_time'])))


################################################################################

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
