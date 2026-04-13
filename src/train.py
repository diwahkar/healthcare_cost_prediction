import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import *
from src.data_preprocessing import load_data, clean_data, create_preprocessor, save_preprocessor
from src.feature_engineering import create_interaction_features, add_cost_tier



warnings.filterwarnings('ignore')


def prepare_features(df):
    '''prepare features for modeling'''
    # create interaction features
    df = create_interaction_features(df)

    # encode categorical variables
    df['sex_male'] = (df['sex'] == 'male').astype(int)
    df['smoker_yes'] = (df['smoker'] == 'yes').astype(int)


    'one-hot encode region'
    region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)
    df = pd.concat([df, region_dummies], axis=1)


    'select features'
    features_cols = [
        'age', 'bmi', 'children',
        'bmi_age_interaction', 'smoker_age', 'children_smoker',
        'sex_male', 'smoker_yes',
        'region_northwest', 'region_southeast', 'region_southwest'
    ]


    X = df[features_cols]
    y = df[TARGET]


    return X, y, features_cols


def train_models(X_train, X_test, y_train, y_test):
    '''train multiple models and return the best'''
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=RANDOM_SEED),
        'XGBoost': XGBRegressor(random_state=RANDOM_SEED, verbosity=0)
    }

    results = {}
    best_model = None
    best_r2 = -np.inf


    for name, model in models.items():
        print(f'\nTraining {name}...')
        model.fit(X_train, y_train)

        # predictions
        y_pred = model.predict(X_test)

        # metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

        print(f'\nMAE: ${mae:,.2f}')
        print(f'\nRMSE: ${rmse:,.3f}')
        print(f'\nR2: {r2:.4f}')

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    print('\nBest model: {best_model.__class__.__name__} with R2 = {best_r2:>4f}')
    return best_model, results



def tune_xgboost(X_train, y_train):
    '''hyperparameter turning for XGBoost'''
    print('\n tuning XGBoost hyperparameters...')

    xgb = XGBRegressor(random_sate= RANDOM_SEED, verbosity=0)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    }


    grid_search = GridSearchCV(
        xgb, param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )


    grid_search.fit(X_train, y_train)

    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best CV R2: {grid_search.best_score_:.4f}')

    return grid_search.best_estimator_


def save_model(model, path=MODEL_PATH):
    '''save trained model'''
    joblib.dump(model, path)


def main():
    print('\nHealthcare Cost Prediction - Training Pipeline')

    # load data
    print('Loading Data...')
    df = load_data()
    df = clean_data(df)

    # feature engineering
    print('Creating Features...')
    X, y, feature_names = prepare_features(df)
    print(f' Features: {feature_names}')
    print(f' X shape: {X.shape}')

    # train-test split
    print('Splitting Data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print(f'   Train: {X_train.shape[0]} samples')
    print(f'    Test: {X_test.shape[0]} samples')


    # train models
    print('Training Models...')
    best_model, results = train_models(X_train, X_test, y_train, y_test)

    # # hyperparameter tuning
    # print('Hyperparameter Tuning...')
    # tuned_xgb = tune_xgboost(X_train, y_train)

    # # evaluate tuned model
    # y_pred_tuned = tuned_xgb.predict(X_test)
    # print(f'\nTuned XGBoost R2: {r2_score(y_test, y_pred_tuned):.4f}')

    # if r2_score(y_test, y_pred_tuned) > results['XGBoost']['r2']:
    #     best_model = tuned_xgb
    #     print('Tuned model is better')


    print('Saving Model...')
    save_model(best_model)

    feature_names_path = MODEL_PATH.parent / 'feature_names.pkl'
    joblib.dump(feature_names, feature_names_path)

    print('\nTraining Complete')

    return best_model, results


if __name__ == '__main__':
    main()
