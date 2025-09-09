import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
from src import regression

# Load raw data
df = pd.read_csv('../data/raw/train.csv')

# Prepare features and target
X = df.drop('SalePrice', axis=1).select_dtypes(include=[np.number])
y = df['SalePrice']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Linear Regression': regression.train_linear_regression(X_train, y_train),
    'Ridge Regression': regression.train_ridge_regression(X_train, y_train),
    'Lasso Regression': regression.train_lasso_regression(X_train, y_train),
    'Decision Tree': regression.train_decision_tree(X_train, y_train),
    'Random Forest': regression.train_random_forest(X_train, y_train),
    'XGBoost': regression.train_xgboost(X_train, y_train),
    'LightGBM': regression.train_lightgbm(X_train, y_train),
    'CatBoost': regression.train_catboost(X_train, y_train),
    'SVR': regression.train_svr(X_train, y_train)
}

# Save models
for name, model in models.items():
    joblib.dump(model, f'../models/{name.replace(" ", "_")}.pkl')

# Evaluate models
results = {}
for name, model in models.items():
    rmse, r2 = regression.evaluate_model(model, X_test, y_test)
    results[name] = {'RMSE': rmse, 'R2': r2}
    print(f'{name}: RMSE = {rmse:.4f}, R2 = {r2:.4f}')

# Compare results
results_df = pd.DataFrame(results).T
print(results_df)

# Save results
results_df.to_csv('../reports/model_results.csv')
