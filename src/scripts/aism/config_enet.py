# Configuration file example for PALLADIO
# version: 2.0

import os
import numpy as np
from palladio import datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from palladio.wrappers import ElasticNetClassifier as ENET

#####################
#   DATASET PATHS ###
#####################

# * All the path are w.r.t. config file path

# The list of all files required for the experiments

data_path = 'dataset_03-2017/data_training.csv'
target_path = 'dataset_03-2017/labels_training.csv'

# pandas.read_csv options
data_loading_options = {
    'delimiter': ',',
    'header': 0,
    'index_col': 0
}
target_loading_options = data_loading_options

try:
    dataset = datasets.load_csv(os.path.join(os.path.dirname(__file__),data_path),
                                os.path.join(os.path.dirname(__file__),target_path),
                                data_loading_options=data_loading_options,
                                target_loading_options=target_loading_options,
                                samples_on='row')
except:
    dataset = datasets.load_csv(os.path.join(os.path.dirname(__file__), 'data'),
                                os.path.join(os.path.dirname(__file__), 'labels'),
                                data_loading_options=data_loading_options,
                                target_loading_options=target_loading_options,
                                samples_on='row')


data, labels = dataset.data, dataset.target
feature_names = dataset.feature_names

#######################
#   SESSION OPTIONS ###
#######################

session_folder = 'mucmd_rfecv-enet'

# The learning task, if None palladio tries to guess it
# [see sklearn.utils.multiclass.type_of_target]
learning_task = None

# The number of repetitions of 'regular' experiments
n_splits_regular = 50

# The number of repetitions of 'permutation' experiments
n_splits_permutation = 50

#######################
#  LEARNER OPTIONS  ###
#######################

# PIPELINE ###
# STEP 1: Imputing
#imp = Imputer(strategy='nn')

# STEP 2: Preprocessing
pp = StandardScaler()

# SETP 3: Classification
clf = ENET()

param_grid = {'classification__l1_ratio': np.linspace(0, 1, 30),
              'classification__alpha': np.logspace(-5, 1, 30)
              }

# COMPOSE THE PIPELINE
pipe = Pipeline([('preproc', pp),
                 ('classification', clf)])

# palladio estimator
estimator = GridSearchCV(pipe, param_grid=param_grid,
                         scoring='accuracy', n_jobs=-1,
                         cv=3, error_score=-np.inf)

# Set options for ModelAssessment
ma_options = {
    'test_size': 0.25,
    'scoring': 'accuracy',
    'n_jobs': -1,
    'n_splits': n_splits_regular,
    'verbose': True
}

# For the Pipeline object, indicate the name of the step from which to
# retrieve the list of selected features
# For a single estimator which has a `coef_` attributes (e.g., elastic net or
# lasso) set to True
vs_analysis = 'classification'

# ~~ Signature Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frequency_threshold = None

# ~~ Plotting Options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
score_surfaces_options = {
    'logspace': ['classification__alpha'],
    'plot_errors': True
}
