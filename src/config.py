import os
from pathlib import Path


# project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'insurance.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'cost_model.pkl'
PREPROCESSOR_PATH = PROJECT_ROOT / 'models' / 'preprocessor.pkl'


'model configuration'
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


'features'
NUMERICAL_FEATURES = ['age', 'bmi', 'children']
CATEGORICAL_FEATURES = ['sex', 'smoker', 'region']
TARGET = 'charges'


'hyperparameters for tuning'
XGB_PARAMS = {
    'n_estimators' : [100, 200, 300],
    'max_depth' : [3, 6, 1],
    'learning_depth': [0.01, 0.05, 0.1],
    'subsample' : [0.8, 1.0]
}


RF_PARAMS = {
    'n_estimators' : [100, 200],
    'max_depth': [10, 20, None],
    'min_sample': [2, 5]
}


'cost tiers (in dollars)'
COST_TIERS = {
    'Low' : (0, 5000),
    'Medium' : (5000, 15000),
    'High' : (15000, 30000),
    'Very High' : (30000, float('inf')),
}


# business constants
AVG_COST_SAVINGS_PER_HIGH_RISK = 5000
