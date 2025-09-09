import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset.
    Parameters:
    - df: pandas DataFrame
    - strategy: 'mean', 'median', 'most_frequent', 'constant'
    Returns:
    - df_imputed: DataFrame with missing values handled
    """
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)
    # For categorical, we can add more logic if needed
    return df_imputed

def encode_categorical(df, method='onehot'):
    """
    Encode categorical variables.
    Parameters:
    - df: pandas DataFrame
    - method: 'onehot' or 'label'
    Returns:
    - encoded data or modified df
    """
    if method == 'onehot':
        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(df.select_dtypes(include=['object']))
        return encoded
    elif method == 'label':
        encoder = LabelEncoder()
        for col in df.select_dtypes(include=['object']):
            df[col] = encoder.fit_transform(df[col])
        return df

def scale_features(df, method='standard'):
    """
    Scale numerical features.
    Parameters:
    - df: pandas DataFrame
    - method: 'standard' or 'minmax'
    Returns:
    - scaled: numpy array
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled

def feature_engineering(df):
    """
    Perform feature engineering.
    Parameters:
    - df: pandas DataFrame
    Returns:
    - df: DataFrame with new features
    """
    # Example: Create age of house
    if 'YearBuilt' in df.columns:
        df['Age'] = 2023 - df['YearBuilt']
    return df
