# The Omni-Analyst: A Comprehensive Machine Learning Suite for Real Estate Valuation and Market Analysis

## Project Description & Objective
 we
The goal of this project is to build a modular Python application that ingests a rich dataset of property listings and uses a wide array of machine learning algorithms to solve different problems within the domain. The objective is not to create one single "best" model, but to demonstrate proficiency in selecting, implementing, and evaluating the right tool for the job across the entire ML spectrum.

**Core Value Proposition:** A real estate agent, investor, or homeowner could use this suite to:
- Predict the sale price of a property (Regression).
- Classify properties into market segments (e.g., Luxury, Affordable, Fixer-Upper).
- Identify similar properties or anomalous listings.
- Gain insights into what features drive property values in a specific area.

## The Dataset

A suitable dataset is crucial. The best choice would be a combination of data from Kaggle (e.g., "House Prices - Advanced Regression Techniques" competition) and public data via an API (like Zillow or Redfin, though API access can be limited). A good dataset should have:
- Numerical Features: Square footage, number of bedrooms/bathrooms, year built, latitude, longitude, price.
- Categorical Features: Neighborhood, zip code, property type (house, condo, townhouse), style.
- Ordinal Features: Condition rating, quality rating.
- Text Data: Property description.
- Dates: Listing date, sale date.
Create a folder 'data' in the project directory and in that folder create sub folder 'raw' and put the different data files in the raw folder.
Please use this dataset as I have used this for my models training: https://www.kaggle.com/datasets/rishitaverma02/house-prices-advanced-regression-techniques 

## Machine Learning Algorithms, Their Uses, and Implementation

### Phase 1: Data Preprocessing & Exploration
Libraries: Pandas, NumPy, Matplotlib, Seaborn.
Tasks: Handling missing values, encoding categorical variables (One-Hot, Label), feature scaling (StandardScaler, MinMaxScaler), feature engineering (e.g., creating "age of house" from "year built"), EDA with visualizations.

### Phase 2: Regression (Predicting Continuous Values)
Problem: Predict the final sale price (SalePrice) of a house.
Algorithms: Linear Regression, Ridge/Lasso Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressors (XGBoost, LightGBM, CatBoost), Support Vector Machines (SVR).
Evaluation Metric: Root Mean Squared Error (RMSE), R-squared.

### Phase 3: Classification (Predicting Categories)
Problem 1: Binary Classification. Create a new target variable Price_Category (e.g., 'Above Median Price' vs. 'Below Median Price').
Problem 2: Multi-Class Classification. Create a new target variable Style_Class (e.g., 'Victorian', 'Ranch', 'Modern').
Algorithms: Logistic Regression, k-Nearest Neighbors, Support Vector Classifier, Naive Bayes, Decision Tree / Random Forest / Gradient Boosting Classifiers.
Evaluation Metric: Accuracy, Precision, Recall, F1-Score, ROC-AUC Curve (for binary).

### Phase 4: Clustering (Finding Hidden Groups)
Problem: Unsupervised learning. Group similar properties together without using the target variable (Price). This can reveal natural market segments.
Algorithms: K-Means Clustering, DBSCAN.
Evaluation Metric: Silhouette Score, Davies-Bouldin Index.

### Phase 5: Natural Language Processing (NLP)
Problem: Use the text from the "Property Description" to improve our price prediction model or to classify properties.
Algorithms: Bag-of-Words / TF-IDF, Sentiment Analysis, Topic Modeling (LDA).

### Phase 6: Dimensionality Reduction
Problem: Reduce high-dimensional data for visualization and modeling.
Algorithms: Principal Component Analysis (PCA), t-SNE / UMAP.

## Project Structure

```
real_state_analysis_and_valuation/
├── data/
│   ├── raw/                    # Original data (train.csv, test.csv, etc.)
│   ├── processed/              # Cleaned and preprocessed data
│   └── external/               # Any additional data (e.g., from APIs)
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Regression_Models.ipynb
│   ├── 03_Classification_Models.ipynb
│   ├── 04_Clustering.ipynb
│   └── 05_NLP_Analysis.ipynb
├── scripts/                    # Executable Python scripts
│   ├── 01_EDA_and_Preprocessing.py
│   ├── 02_Regression_Models.py
│   ├── 03_Classification_Models.py
│   ├── 04_Clustering.py
│   └── 05_NLP_Analysis.py
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── preprocessing.py        # Data preprocessing functions
│   ├── regression.py           # Regression model training and evaluation
│   └── visualization.py        # Plotting and visualization functions
├── models/                     # Saved trained models (.pkl files)
├── reports/                    # Generated reports and summaries
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview, setup instructions, and results summary
```

## Setup Instructions

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Place your dataset in the `data/raw/` directory (e.g., train.csv, test.csv from Kaggle's House Prices dataset).
4. Run the scripts in order for data preprocessing, model training, and analysis:
   - `python scripts/01_EDA_and_Preprocessing.py` for exploratory data analysis.
   - `python scripts/02_Regression_Models.py` for training and evaluating regression models.
   - Alternatively, use the Jupyter notebooks in the `notebooks/` directory for interactive analysis.

## Key Python Libraries

- Core: pandas, numpy
- Visualization: matplotlib, seaborn, plotly
- ML Algorithms: scikit-learn
- Advanced ML: xgboost, lightgbm, catboost
- NLP: nltk, textblob, spacy
- Statistics: scipy
- Hyperparameter Tuning: scikit-learn (GridSearchCV, RandomizedSearchCV)

## Results Summary

### Phase 1: EDA and Preprocessing
- Dataset: test.csv (1459 entries, 80 columns)
- Key statistics: Mean LotArea = 9819, Mean YearBuilt = 1971, etc.
- Missing values handled, categorical encoding, and scaling applied.

### Phase 2: Regression Models
Trained and evaluated multiple regression models on the Ames Housing dataset:

| Model              | RMSE       | R² Score  |
|--------------------|------------|-----------|
| Linear Regression  | 36872.91   | 0.8227    |
| Ridge Regression   | 36872.03   | 0.8228    |
| Lasso Regression   | 36872.59   | 0.8227    |
| Decision Tree      | 39227.35   | 0.7994    |
| Random Forest      | 30061.48   | 0.8822    |
| XGBoost            | 29070.29   | 0.8898    |
| LightGBM           | 29978.93   | 0.8828    |
| CatBoost           | 28764.92   | 0.8921    |
| SVR                | 88652.43   | -0.0246   |

Best performing model: CatBoost (RMSE: 28764.92, R²: 0.8921)

### Phase 3: Classification Models
Binary classification on price categories:

| Model              | Accuracy | F1 Score |
|--------------------|----------|----------|
| Logistic Regression| 0.830    | 0.8426   |
| k-NN               | 0.800    | 0.8095   |
| SVM                | 0.830    | 0.8365   |
| Naive Bayes        | 0.810    | 0.8173   |
| Decision Tree      | 0.840    | 0.8545   |
| Random Forest      | 0.885    | 0.8930   |

Best performing model: Random Forest (Accuracy: 0.885, F1: 0.8930)

### Phase 4: Clustering
Unsupervised clustering on property features:

- K-Means Silhouette Score: 0.6773
- DBSCAN Silhouette Score: 0.5481

K-Means performs better for this dataset.

### Phase 5: NLP Analysis
Text analysis on property descriptions:

- TF-IDF Matrix: 5 samples, 33 features
- Sentiment Analysis: Positive sentiments ranging from 0.00 to 0.53
- LDA Topics: 2 topics identified (e.g., historic features vs. modern amenities)

### Key Insights
- Advanced tree-based models (CatBoost, XGBoost) outperform linear models for price prediction.
- Random Forest is effective for classification tasks.
- K-Means clustering reveals distinct property segments.
- NLP can extract meaningful features from property descriptions for enhanced modeling.
