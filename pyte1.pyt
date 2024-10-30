import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
df = pd.read_csv('fetal_birth_weight.dat')

# Split the data into features and target variable
X = df.drop(columns=['birth_weight'])
y = df['birth_weight']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM, CatBoost, and GradientBoosting regressors
lgbm_model = LGBMRegressor()
catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=0)
gb_model = GradientBoostingRegressor()

# Hyperparameter tuning for LightGBM
lgbm_params = {
    'num_leaves': [20, 30, 40],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}
lgbm_random = RandomizedSearchCV(estimator=lgbm_model, param_distributions=lgbm_params, n_iter=10, cv=3, random_state=42)
lgbm_random.fit(X_train, y_train)
best_lgbm_model = lgbm_random.best_estimator_

# Hyperparameter tuning for CatBoost
catboost_params = {
    'iterations': [1000, 1500, 2000],
    'learning_rate': [0.05, 0.1, 0.2],
    'depth': [4, 6, 8]
}
catboost_random = RandomizedSearchCV(estimator=catboost_model, param_distributions=catboost_params, n_iter=10, cv=3, random_state=42)
catboost_random.fit(X_train, y_train)
best_catboost_model = catboost_random.best_estimator_

# Train GradientBoostingRegressor for comparison
gb_model.fit(X_train, y_train)

# Save the best models to pickle files
with open('best_lgbm_model.pkl', 'wb') as f:
    pickle.dump(best_lgbm_model, f)

with open('best_catboost_model.pkl', 'wb') as f:
    pickle.dump(best_catboost_model, f)

# Make predictions with the best models
lgbm_pred = best_lgbm_model.predict(X_test)
catboost_pred = best_catboost_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Evaluate the models
lgbm_rmse = mean_squared_error(y_test, lgbm_pred, squared=False)
catboost_rmse = mean_squared_error(y_test, catboost_pred, squared=False)
gb_rmse = mean_squared_error(y_test, gb_pred, squared=False)

print(f"Best LightGBM RMSE: {lgbm_rmse}")
print(f"Best CatBoost RMSE: {catboost_rmse}")
print(f"GradientBoostingRegressor RMSE: {gb_rmse}")
