# Week 3 Task: Advanced Data Analysis and Visualization in Logistics
# Project: E-Commerce Delivery Optimization
#
# This script:
# 1. Loads the Week 3 logistics dataset
# 2. Performs Exploratory Data Analysis (EDA)
# 3. Calculates descriptive statistics and correlations
# 4. Creates 8 visualizations
# 5. Saves all visualization outputs in the Week_3 folder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "week3_logistics_dataset.csv"
OUTPUT_DIR = BASE_DIR

# Check whether the dataset exists
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Make sure week3_logistics_dataset.csv is in the same folder as this script."
    )


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("WEEK 3 - ADVANCED DATA ANALYSIS AND VISUALIZATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ---------------------------------------------------------
# 3. Descriptive Statistics
# ---------------------------------------------------------

numeric_columns = [
    "Distance_km",
    "Delivery_Time_hr",
    "Shipment_Volume",
    "Package_Weight_kg",
    "Transportation_Cost"
]

print("\nDescriptive Statistics:")
print(df[numeric_columns].describe())

print("\nMean:")
print(df[numeric_columns].mean())

print("\nMedian:")
print(df[numeric_columns].median())

print("\nStandard Deviation:")
print(df[numeric_columns].std())


# ---------------------------------------------------------
# 4. Correlation Analysis
# ---------------------------------------------------------

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)


# ---------------------------------------------------------
# 5. Group Analysis
# ---------------------------------------------------------

print("\nAverage Metrics by Vehicle Type:")
vehicle_analysis = df.groupby("Vehicle_Type")[
    ["Delivery_Time_hr", "Transportation_Cost"]
].mean().round(2)

print(vehicle_analysis)

print("\nAverage Delivery Time by Region:")
region_analysis = (
    df.groupby("Region")["Delivery_Time_hr"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)

print(region_analysis)

print("\nDelivery Status Counts:")
print(df["Delivery_Status"].value_counts())

print("\nWeather-wise Delivery Status:")
weather_status = pd.crosstab(
    df["Weather"],
    df["Delivery_Status"]
)

print(weather_status)


# ---------------------------------------------------------
# 6. Visualization 1
# Delivery Time Distribution
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Delivery_Time_hr"],
    bins=20,
    edgecolor="black"
)

plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (hours)")
plt.ylabel("Number of Deliveries")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "viz1_delivery_time_distribution.png",
    dpi=180,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 7. Visualization 2
# Transportation Cost Distribution
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Transportation_Cost"],
    bins=20,
    edgecolor="black"
)

plt.title("Transportation Cost Distribution")
plt.xlabel("Transportation Cost")
plt.ylabel("Number of Shipments")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "viz2_transport_cost_distribution.png",
    dpi=180,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 8. Visualization 3
# Distance vs Delivery Time
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Distance_km"],
    df["Delivery_Time_hr"],
    alpha=0.65
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (hours)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "viz3_distance_vs_delivery.png",
    dpi=180,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 9. Visualization 4
# Distance vs Transportation Cost
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Distance_km"],
    df["Transportation_Cost"],
    alpha=0.65
)

plt.title("Distance vs Transportation Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "viz4_distance_vs_cost.png",
    dpi=180,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 10. Visualization 5
# Average Transportation Cost by Vehicle Type
# ---------------------------------------------------------

vehicle_cost = (
    df.groupby("Vehicle_Type")["Transportation_Cost"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

vehicle_cost.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Average Transportation Cost by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Average Transportation Cost")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "viz5_cost_by_vehicle.png",
    dpi=180,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 11. Visualization 6
# Average Delivery Time by Region
# ---------------------------------------------------------

region_time = (
    df.groupby("Region")["Delivery_Time_hr"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 5))

region_time.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Average Delivery Time by Region")
plt.xlabel("Region")
plt.ylabel("Average Delivery Time (hours)")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "viz6_delivery_time_by_region.png",
    dpi=180,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 12. Visualization 7
# On-Time vs Delayed Deliveries
# ---------------------------------------------------------

status_counts = df["Delivery_Status"].value_counts()

plt.figure(figsize=(8, 5))

status_counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("On-Time vs Delayed Deliveries")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Deliveries")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "viz7_delivery_status.png",
    dpi=180,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 13. Visualization 8
# Correlation Heatmap
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.index)),
    correlation.index
)

plt.title("Correlation Heatmap of Logistics Metrics")

# Add correlation values inside the heatmap
for i in range(len(correlation.index)):
    for j in range(len(correlation.columns)):
        plt.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "viz8_correlation_heatmap.png",
    dpi=180,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 14. Key Insights
# ---------------------------------------------------------

distance_delivery_corr = correlation.loc[
    "Distance_km",
    "Delivery_Time_hr"
]

distance_cost_corr = correlation.loc[
    "Distance_km",
    "Transportation_Cost"
]

weight_cost_corr = correlation.loc[
    "Package_Weight_kg",
    "Transportation_Cost"
]

print("\n" + "=" * 60)
print("KEY ANALYTICAL INSIGHTS")
print("=" * 60)

print(
    f"\n1. Distance and delivery time correlation: "
    f"{distance_delivery_corr:.2f}"
)

print(
    f"2. Distance and transportation cost correlation: "
    f"{distance_cost_corr:.2f}"
)

print(
    f"3. Package weight and transportation cost correlation: "
    f"{weight_cost_corr:.2f}"
)

print(
    f"4. Slowest region based on average delivery time: "
    f"{region_analysis.index[0]} "
    f"({region_analysis.iloc[0]:.2f} hours)"
)

print(
    f"5. Fastest region based on average delivery time: "
    f"{region_analysis.index[-1]} "
    f"({region_analysis.iloc[-1]:.2f} hours)"
)

print(
    f"6. Total on-time deliveries: "
    f"{status_counts.get('On Time', 0)}"
)

print(
    f"7. Total delayed deliveries: "
    f"{status_counts.get('Delayed', 0)}"
)


# ---------------------------------------------------------
# 15. Final Recommendations
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

print("""
1. Optimize routes for long-distance deliveries.
2. Select vehicles according to shipment volume and package weight.
3. Monitor regions with higher average delivery times.
4. Consider weather conditions when planning delivery schedules.
5. Track transportation cost per shipment and per kilometre.
6. Monitor delayed deliveries regularly using dashboards.
7. Apply the same analysis to real historical logistics data.
""")


print("\nAnalysis completed successfully.")
print("All 8 visualizations have been saved in the Week_3 folder.")
