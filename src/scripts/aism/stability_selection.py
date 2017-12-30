#!/usr/bin/env python
"""Perform selection stability on a bunch of models on the AISM dataset."""
from __future__ import print_function

import warnings

import cPickle as pkl
import numpy as np

from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Re-use the utilities of the challenge script
from aging_model_challenge import load_raw_data
from aging_model_challenge import preprocess
from aging_model_challenge import dump


def evaluate(estimator, X, y):
    """Compute the selected metrics on the input (test) set."""
    y_pred = estimator.predict(X)
    # metrics
    r2 = metrics.r2_score(y, y_pred)
    mae = metrics.mean_absolute_error(y, y_pred)
    ev = metrics.explained_variance_score(y, y_pred)
    mse = metrics.mean_squared_error(y, y_pred)
    return {'R2': r2, 'MAE': mae, 'EV': ev, 'MSE': mse}


def stability_selection(estimator, X, y):
    """Naive stability selection process with MC samples."""
    rs = ShuffleSplit(n_splits=500, test_size=.25)
    coefs = []
    test_metrics = []
    print('- Running {} splits:'.format(rs.n_splits))
    for i, (train_index, test_index) in enumerate(rs.split(X)):
        estimator.fit(X.iloc[train_index], y.iloc[train_index])
        coefs.append(estimator.best_estimator_.steps[1][1].coef_)
        test_metrics.append(evaluate(estimator, X.iloc[test_index], y.iloc[test_index]))
        print('\t* Split {}/{} done.'.format(i+1, rs.n_splits))
    return coefs, test_metrics


def main():
    """Stability selection main routine."""
    print('--------------------------------------')
    print('Stability selection with Lasso')
    print('--------------------------------------\n')

    # 0. Load data
    df = load_raw_data()
    features = df.columns.drop('age')
    print('- Raw data loaded.')

    # 1. Split data/labels
    raw_data = df[features]
    labels = df['age']
    print('- {} samples / {} features.'.format(*raw_data.shape))

    # 2. Preprocess raw data
    data = preprocess(raw_data, poly_feat=True)
    data = data.drop('gender^2', axis=1) # drop nonsense feature

    # 3. Define the cross-validated pipeline
    pipe = Pipeline([['preproc', StandardScaler()],
                     ['predict', Lasso()]])
    pipe_cv = GridSearchCV(pipe, param_grid={'predict__alpha': np.logspace(-3, 2, 30)},
                           n_jobs=-1)

    coefs, scores = stability_selection(pipe_cv, data, labels)

    print('- Dumping results on disk ...')
    dump(coefs, 'ss_lasso_coefs.pkl')
    dump(scores, 'ss_lasso_scores.pkl')
    print('done.')

################################################################################

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
