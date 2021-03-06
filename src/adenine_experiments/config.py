#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration file for adenine."""

from adenine.utils import data_source

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = '_small'
output_root_folder = 'results'
plotting_context = 'notebook'  # one of {paper, notebook, talk, poster}
file_format = 'pdf'  # or 'png'
use_compression = False  # use gzip to compress the results

# ----------------------------  INPUT DATA ---------------------------- #
# Load an example dataset or specify your input data in tabular format
data_file = '../data../AISM/anonimyzed_data_12-2017.csv'
labels_file = '../data../AISM/anonimyzed_labels_12-2017.csv'  # OPTIONAL
samples_on = 'rows'  # if samples lie on columns use 'cols' or 'col'
data_sep = ','  # the data separator. e.g., ',', '\t', ' ', ...
X, y, feat_names, index = data_source.load('custom',
                                           data_file, labels_file,
                                           samples_on=samples_on,
                                           n_samples=150,
                                           sep=data_sep)

# -----------------------  PIPELINES DEFINITION ------------------------ #
# --- Missing values imputing --- #
step0 = {'Impute': [True, {'missing_values': 'NaN',
                            'strategy': ['median']}]}

# --- Data preprocessing --- #
step1 = {'None': [False], 'Recenter': [False], 'Standardize': [True],
         'Normalize': [False, {'norm': ['l1', 'l2']}],
         'MinMax': [True, {'feature_range': [(0, 1)]}]}

# --- Unsupervised features learning --- #
# affinity ca be precumputed for SE
step2 = {'PCA': [True, {'n_components': 3}],
         'IncrementalPCA': [False],
         'RandomizedPCA': [False],
         'KernelPCA': [False, {'kernel': ['linear', 'rbf', 'poly']}],
         'Isomap': [True, {'n_neighbors': 5}],
         'LLE': [False, {'n_neighbors': 5,
                         'method': ['standard', 'modified',
                                    'hessian', 'ltsa']}],
         'SE': [False, {'affinity': ['nearest_neighbors', 'rbf']}],
         'MDS': [False, {'metric': True}],
         'tSNE': [False, {'perplexity': [100]}],
         'RBM': [False, {'n_components': 256}],
         'None': [False]
         }

# --- Clustering --- #
# affinity ca be precumputed for AP, Spectral and Hierarchical
step3 = {'KMeans': [True, {'n_clusters': [2, 'auto']}],
         'AP': [False, {'preference': ['auto']}],
         'MS': [False],
         'Spectral': [False, {'n_clusters': [3]}],
         'Hierarchical': [True, {'n_clusters': [3],
                                  'affinity': ['euclidean'],
                                  'linkage':  ['ward']}]
         }
