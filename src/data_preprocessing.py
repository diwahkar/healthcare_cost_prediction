import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import *



def load_data():
    '''load insurance dataset'''
    df = pd.read_csv(DATA_PATH)
    print(f'Data loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    return df


def clean_data(df):
    '''basci data cleaning'''
    if df.isnull().sum().sum()> 0:
        print(f'Missing valus found: {df.isnull().sum()}')
        df = df.dropna()


    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f'Dropping {duplicates} duplicate rows')
        df = df.drop_duplicates()

    return df


def create_preprocessor():
    '''create preprocessing pipeline'''
    numeric_transformer = StandardScaler() # z-normalization
    categorical_transfomer = OneHotEncoder(drop='first', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformer = [
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transfomer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


def save_preprocessor(preprocessor, path=PREPROCESSOR_PATH):
    '''save preprocess to disk'''
    joblib.dump(preprocessor, path)
    print(f'Preprocessor saved to {path}')


def load_preprocessor(path=PREPROCESSOR_PATH):
    '''load preprocess from disk'''
    return joblib.load(path)



if __name__ == '__main__':
    df = load_data()
    df = clean_data()
    preprocessor = create_preprocessor()
    print('Preprocessing pipeline created successfully')
