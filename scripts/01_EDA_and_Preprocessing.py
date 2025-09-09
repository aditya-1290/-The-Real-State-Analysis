
#!/usr/bin/env python

# EDA and Preprocessing
# This python script handles Exploratory Data Analysis (EDA) and data preprocessing for the real estate dataset.


import sys
sys.path.append('.')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import preprocessing, visualization

# Load raw data
data_path = '../data/raw/test.csv'  # Update with your actual filename
df = pd.read_csv(data_path)

# Basic info
print(df.info())
print(df.describe())

# Visualize distributions
for col in ['LotArea', 'YearBuilt']:
    visualization.plot_histogram(df, col)

# Handle missing values
df_num = df.select_dtypes(include=['number'])
df_num_imputed = preprocessing.handle_missing_values(df_num, strategy='mean')

# Feature engineering
df = preprocessing.feature_engineering(df)

# Encode categorical variables
df_cat = df.select_dtypes(include=['object'])
df_cat_encoded = preprocessing.encode_categorical(df_cat, method='onehot')

# Scale numerical features
df_num_scaled = preprocessing.scale_features(df_num_imputed, method='standard')

# Combine processed features as needed for modeling
# ...

print('Preprocessing complete.')
