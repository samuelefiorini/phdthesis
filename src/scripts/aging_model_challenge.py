#!/usr/bin/env python
"""Challenge different ML methods on the aging problem."""

from __future__ import print_function

import os
import time
import warnings

import cPickle as pkl
import numpy as np
import pandas as pd

from fancyimpute import KNN
from hurry.filesize import size
from palladio.model_assessment import ModelAssessment
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def load_raw_data():
    """Load the raw data of the aging problem via pandas."""
    return pd.read_csv(os.path.join('..', 'data', 'aging_data.csv'),
                       header=0, index_col=0)

def preprocess(dfx, poly_feat=False):
    """Categorical feature mapping + KNN imputing.

    dfx: input data DataFrame object with shape (n,d)
    """
    # 1. Convert gender [m/f] in [0,1]
    dfx.loc[:, 'gender'] = dfx['gender'].apply(lambda x: int(x == 'f'))

    # 2. Impute missing values via KNN
    dfx = pd.DataFrame(data=KNN(k=3).complete(dfx.values),
                       index=dfx.index, columns=dfx.columns)

    # 3. If needed, compute polynomial features
    if poly_feat:
        polyfeat = PolynomialFeatures(degree=2, include_bias=True)
        dfx_values = polyfeat.fit_transform(dfx.values)
        dfx_columns = polyfeat.get_feature_names(dfx.columns)
        dfx = pd.DataFrame(data=dfx_values,
                           index=dfx.index,
                           columns=dfx_columns)

    return dfx

def create_pipelines():
    """Create all the models organized in a dictionary."""

    names = [
        'gradient_boosting',
        'random_forests',
        'elasticnet',
        'lasso',
        'linear_regression',
        'kernel_ridge',
        'ridge',
        'mlp',
        'rbf_svr',
        'linear_svr'
        ]

    estimators = [
        GradientBoostingRegressor(),
        RandomForestRegressor(),
        ElasticNet(),
        Lasso(),
        LinearRegression(),
        KernelRidge(),
        Ridge(),
        MLPRegressor(),
        SVR(),
        SVR()
        ]

    params = [
        {'predict__max_depth': np.arange(3, 20), # gradient_boosting
         'predict__n_estimators': np.arange(1, 103, 3)},
        {'predict__max_features': np.arange(3, 12), # random_forests
         'predict__n_estimators': [500]},
        {'predict__alpha': np.logspace(-3, 2, 30), # elastic_net
         'predict__l1_ratio': np.linspace(1e-3, 1, 30)},
        {'predict__alpha': np.logspace(-3, 2, 30)}, # lasso
        {}, # linear_regression
        {'predict__alpha': np.logspace(-3, 2, 30), # kernel ridge
         'predict__kernel': ['rbf'],
         'predict__gamma': np.logspace(-3, 2, 30)},
        {'predict__alpha': np.logspace(-3, 2, 30)}, # ridge
        {'predict__alpha': np.logspace(-5, 2, 20), # mlp
         'predict__hidden_layer_sizes': [2**i for i in range(12)]},
        {'predict__C': np.logspace(-1, 3, 30), # rbf svr
         'predict__kernel': ['rbf'],
         'predict__gamma': np.logspace(-3, 2, 30)},
        {'predict__C': np.logspace(-1, 3, 30), # linear svr
         'predict__kernel': ['linear']}
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
    print('The aging problem model challenge')
    print('--------------------------------------\n')

    # 0. Load data
    df = load_raw_data()
    features = df.columns.drop('age')
    print('- Raw data loaded.')

    # 1. Split data/labels
    raw_data = df[features]
    labels = df['age']
    print('- {} samples / {} features.'.format(*raw_data.shape))

    # 2. Create pipelines
    pipes = create_pipelines()
    print('- {} pipelines created'.format(len(pipes)))

    # 3. Perform the model challenge twice, one for linear
    # and the other for polynomial features
    for toggle_poly_feat in [True, False]:
        print('\n----- POLYNOMIAL FEATURES {} -----'.format('ON' if toggle_poly_feat else 'OFF'))

        # 3.1 Preprocess data
        data = preprocess(raw_data, poly_feat=toggle_poly_feat)
        print('- preprocessing done.')
        print('- {} samples / {} features.\n'.format(*data.shape))

         # use a different name for the two versions of the dataset
        tail = 'poly' if toggle_poly_feat else ''

        # Individually evaluate the performance of each pipeline
        times = {}
        fitted_ma = {}
        for pipe in pipes.keys():
            print('Fitting {}...'.format(pipe))
            ma = ModelAssessment(estimator=pipes[pipe],
                                 cv=ShuffleSplit(n_splits=100, test_size=.25),
                                 scoring='neg_mean_absolute_error',
                                 n_jobs=-1)

            # Fit the model evaluation object
            tic = time.time()
            ma.fit(data.values, labels.values.ravel())
            toc = time.time()

            times[pipe] = toc - tic # save time
            dump(times, 'times_'+tail+'.pkl')
            fitted_ma[pipe] = ma # save results
            dump(fitted_ma, 'fitted_ma_'+tail+'.pkl')
            print('Done in {:3.5} sec.\n'.format(times[pipe]))


################################################################################

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
