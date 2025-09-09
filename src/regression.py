from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train, alpha=1.0):
    """
    Train a Ridge Regression model.
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_lasso_regression(X_train, y_train, alpha=1.0):
    """
    Train a Lasso Regression model.
    """
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree Regressor.
    """
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Regressor.
    """
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost Regressor.
    """
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    """
    Train a LightGBM Regressor.
    """
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    return model

def train_catboost(X_train, y_train):
    """
    Train a CatBoost Regressor.
    """
    model = CatBoostRegressor(verbose=0)
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train):
    """
    Train a Support Vector Regressor.
    """
    model = SVR()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using RMSE and R-squared.
    """
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2
