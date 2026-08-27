import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans


# Load dataset
df = pd.read_csv("logistics_data.csv")

# Basic data exploration
print(df.head())
print(df.info())
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Exploratory Data Analysis
plt.figure(figsize=(8, 5))
sns.histplot(df["Actual Delivery Time"])
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time")
plt.ylabel("Number of Deliveries")
plt.show()


# Regression: Delivery Time Prediction
X = df[["Distance", "Package Weight"]]
y = df["Actual Delivery Time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


# Clustering: Identify Delivery Patterns
features = df[[
    "Distance",
    "Delivery Cost",
    "Actual Delivery Time"
]]

kmeans = KMeans(n_clusters=3, random_state=42)

df["Cluster"] = kmeans.fit_predict(features)

print(df[[
    "Distance",
    "Delivery Cost",
    "Actual Delivery Time",
    "Cluster"
]].head())
