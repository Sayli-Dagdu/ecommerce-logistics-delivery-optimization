# Week 4 Task: Predictive Modeling and Optimization in Logistics Systems

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("week4_logistics_dataset.csv")

# Target and features
target = "Delivery_Time_hr"
features = [
    "Region", "Vehicle_Type", "Distance_km", "Shipment_Volume",
    "Package_Weight_kg", "Transportation_Cost", "Weather"
]

X = df[features]
y = df[target]

categorical_features = ["Region", "Vehicle_Type", "Weather"]
numeric_features = [
    "Distance_km", "Shipment_Volume",
    "Package_Weight_kg", "Transportation_Cost"
]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Linear Regression
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# Random Forest
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluation
def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

evaluate_model("Linear Regression", y_test, linear_pred)
evaluate_model("Random Forest", y_test, rf_pred)

# 5-fold cross-validation
cv_scores = cross_val_score(
    rf_model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
print("5-fold CV MAE:", -cv_scores.mean())

# Hyperparameter tuning
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_model, param_grid, cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test)

evaluate_model("Tuned Random Forest", y_test, best_pred)
print("Best parameters:", grid_search.best_params__)

# Business use:
# Use predicted delivery time to flag high-risk shipments,
# prioritize route optimization, allocate suitable vehicles,
# and add time buffers for adverse weather conditions.
