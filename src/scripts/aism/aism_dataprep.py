#!/usr/bin/env python
"""Challenge different ML methods on the aging problem."""

from __future__ import print_function

import os
import warnings

import cPickle as pkl
import numpy as np
import pandas as pd

from fancyimpute import KNN
from hurry.filesize import size
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder

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
    idx = {}
    for i in range(1, 10): # S01 - S09
        idx[i] = filter(lambda x: x.endswith('S0'+str(i)), data.index)
    idx[10] = filter(lambda x: x.endswith('S10'), data.index)
    idx[11] = filter(lambda x: x.endswith('S11'), data.index)

    idx_train = flatten([idx[i] for i in [1, 2, 3]])
    idx_valid = idx[4]
    idx_test = flatten([idx[i] for i in range(5, 12)])

    return idx_train, idx_valid, idx_test


def preprocess(data, labels):
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

    # Update the data matrix
    data = pd.concat([data.drop(edi_items, axis=1),
                      pd.DataFrame(xx, columns=ohe_edi_items, index=data.index)
                      ], axis=1)

    # 3. Keep only RR and SP
    rrsp = np.where(np.logical_or(labels.values == 'RR', labels.values == 'SP'))[0]

    return data.iloc[rrsp], labels.iloc[rrsp]


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
    data, labels = preprocess(data, labels)
    print('- Data preprocessed: {} samples / {} features.'.format(*data.shape))

    # 2. Train/validation/test split
    idx_train, idx_valid, idx_test = training_validation_test_split(data)
    print('     * {} training samples'.format(len(idx_train)))
    print('     * {} validation samples'.format(len(idx_valid)))
    print('     * {} test samples'.format(len(idx_test)))

    # 3. Save data matrices
    # 3.1 Training
    pd.DataFrame(data=data.loc[idx_train], columns=data.columns,
                 index=idx_train).to_csv('data_training.csv')
    pd.DataFrame(data=labels.loc[idx_train], columns=labels.columns,
                 index=idx_train).to_csv('labels_training.csv')
    print('- Training data saved')

    # 3.2 Validation
    pd.DataFrame(data=data.loc[idx_valid], columns=data.columns,
                 index=idx_valid).to_csv('data_valid.csv')
    pd.DataFrame(data=labels.loc[idx_valid], columns=labels.columns,
                 index=idx_valid).to_csv('labels_valid.csv')
    print('- Validation data saved')

    # 3.3 Test
    pd.DataFrame(data=data.loc[idx_test], columns=data.columns,
                 index=idx_test).to_csv('data_test.csv')
    pd.DataFrame(data=labels.loc[idx_test], columns=labels.columns,
                 index=idx_test).to_csv('labels_test.csv')
    print('- Test data saved')

################################################################################

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
