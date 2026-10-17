
"""
Cognifyz Data Science Internship
Level 3 - Task 1: Predictive Modeling

Objective:
Predict restaurant Aggregate rating using regression models.
Compare Linear Regression, Decision Tree, and Random Forest.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# 1. Set up project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load and inspect the dataset
# --------------------------------------------------

print("=" * 65)
print("COGNIFYZ INTERNSHIP - LEVEL 3, TASK 1")
print("RESTAURANT RATING PREDICTION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. Clean the target variable
# --------------------------------------------------

target = "Aggregate rating"

if target not in df.columns:
    raise ValueError(f"Required target column '{target}' not found.")

df[target] = pd.to_numeric(df[target], errors="coerce")

# In this dataset, rating 0 means "Not rated".
# Exclude unrated records rather than treating 0 as a
# genuine restaurant rating.
df = df.dropna(subset=[target])
df = df[df[target] > 0].copy()

print("\nDataset shape after excluding unrated restaurants:")
print(df.shape)

print("\nTarget rating statistics:")
print(df[target].describe())


# --------------------------------------------------
# 4. Select features and prevent target leakage
# --------------------------------------------------

# Rating color and Rating text are derived from the rating.
# Including them would leak information about the target.
# Restaurant identifiers and names are also excluded.

excluded_columns = [
    "Restaurant ID",
    "Restaurant Name",
    "Address",
    "Locality",
    "Locality Verbose",
    "Currency",
    "Rating color",
    "Rating text",
    target,
]

X = df.drop(
    columns=excluded_columns,
    errors="ignore"
).copy()

y = df[target].copy()

# Convert known numeric columns to numeric values.
numeric_candidates = [
    "Country Code",
    "Longitude",
    "Latitude",
    "Average Cost for two",
    "Price range",
    "Votes",
]

for column in numeric_candidates:
    if column in X.columns:
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        )

# Treat country codes and price ranges as categories.
# They are labels/groups, not continuous measurements.
for column in ["Country Code", "Price range"]:
    if column in X.columns:
        X[column] = X[column].astype("str")

# Identify numeric and categorical feature columns.
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

print("\nFeatures used:", X.columns.tolist())
print("\nNumeric features:", numeric_features)
print("\nCategorical features:", categorical_features)
print("\nTarget leakage columns excluded:", [
    "Rating color",
    "Rating text",
])


# --------------------------------------------------
# 5. Split into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 6. Build preprocessing pipeline
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ],
    remainder="drop",
)


# --------------------------------------------------
# 7. Define regression algorithms
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ),
}


# --------------------------------------------------
# 8. Train and evaluate each model
# --------------------------------------------------

results = []
predictions = {}
trained_models = {}

for model_name, regressor in models.items():

    print(f"\nTraining {model_name}...")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
    })

    predictions[model_name] = y_pred
    trained_models[model_name] = model

    print(f"MAE:      {mae:.4f}")
    print(f"MSE:      {mse:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")


# --------------------------------------------------
# 9. Compare model performance
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True,
)

results_path = OUTPUT_DIR / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 65)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))

# Lowest RMSE is the primary selection criterion.
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_predictions = predictions[best_model_name]

print("\nBest model based on test RMSE:", best_model_name)


# --------------------------------------------------
# 10. Save actual vs predicted values
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "Actual Rating": y_test.to_numpy(),
    "Predicted Rating": best_predictions,
})

prediction_df.to_csv(
    OUTPUT_DIR / "actual_vs_predicted.csv",
    index=False,
)


# --------------------------------------------------
# 11. Plot model comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["RMSE"],
)

plt.title("Regression Model Comparison - RMSE")
plt.xlabel("Regression Model")
plt.ylabel("Root Mean Squared Error (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# 12. Plot actual vs predicted ratings
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.4,
)

rating_min = min(y_test.min(), best_predictions.min())
rating_max = max(y_test.max(), best_predictions.max())

plt.plot(
    [rating_min, rating_max],
    [rating_min, rating_max],
    linestyle="--",
)

plt.title(f"Actual vs Predicted Ratings - {best_model_name}")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# 13. Save a text report
# --------------------------------------------------

report_path = OUTPUT_DIR / "task1_report.txt"

with open(report_path, "w", encoding="utf-8") as report:

    report.write("COGNIFYZ DATA SCIENCE INTERNSHIP\n")
    report.write("LEVEL 3 - TASK 1: PREDICTIVE MODELING\n\n")

    report.write(f"Original dataset shape: {pd.read_csv(DATA_PATH).shape}\n")
    report.write(f"Rated restaurants used: {len(df)}\n")
    report.write(f"Training records: {len(X_train)}\n")
    report.write(f"Testing records: {len(X_test)}\n\n")

    report.write("Target: Aggregate rating\n")
    report.write(
        "Records with rating 0 were excluded because they "
        "represent restaurants marked as Not rated.\n"
    )
    report.write(
        "Rating color and Rating text were excluded to avoid "
        "target leakage.\n\n"
    )

    report.write("MODEL PERFORMANCE COMPARISON\n")
    report.write(results_df.to_string(index=False))
    report.write("\n\n")

    report.write(
        f"Best model by test RMSE: {best_model_name}\n"
    )
    report.write(
        "Interpretation: lower MAE and RMSE indicate smaller "
        "prediction errors. Higher R2 indicates more variance "
        "explained on the test set.\n"
    )
    report.write(
        "These metrics measure performance on this held-out "
        "test split and do not guarantee performance on new data.\n"
    )

print("\nAll outputs saved to:", OUTPUT_DIR.resolve())
print("\nTask 1 training and evaluation completed.")
