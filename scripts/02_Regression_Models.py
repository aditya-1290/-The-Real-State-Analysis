import pandas as pd
from sklearn.model_selection import train_test_split
from src import regression

# Load processed data (assuming from previous notebook)
# df_processed = pd.read_csv('../data/processed/processed_data.csv')
# X = df_processed.drop('SalePrice', axis=1)
# y = df_processed['SalePrice']

# For demonstration, using sample data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

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

# Evaluate models
results = {}
for name, model in models.items():
    rmse, r2 = regression.evaluate_model(model, X_test, y_test)
    results[name] = {'RMSE': rmse, 'R2': r2}
    print(f'{name}: RMSE = {rmse:.4f}, R2 = {r2:.4f}')

# Compare results
results_df = pd.DataFrame(results).T
print(results_df)
