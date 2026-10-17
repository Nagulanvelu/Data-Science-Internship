
# ============================================================
# Cognifyz Data Science Internship
# Level 1 - Task 1: Data Exploration and Preprocessing
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up output directory
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------

df = pd.read_csv("Dataset.csv")

print("=" * 60)
print("DATA EXPLORATION AND PREPROCESSING")
print("=" * 60)

# ------------------------------------------------------------
# 2. Identify the number of rows and columns
# ------------------------------------------------------------

print("\n1. DATASET DIMENSIONS")
print("-" * 40)

print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

# ------------------------------------------------------------
# 3. Check missing values in each column
# ------------------------------------------------------------

print("\n2. MISSING VALUE ANALYSIS")
print("-" * 40)

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("No missing values found.")

print("\nTotal missing values:", missing_values.sum())

# ------------------------------------------------------------
# 4. Handle missing values
# ------------------------------------------------------------

print("\n3. HANDLING MISSING VALUES")
print("-" * 40)

# Fill missing cuisine values with Unknown.
# This preserves restaurant records without inventing cuisine data.
if "Cuisines" in df.columns:
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# Fill other missing values according to their data type.
# Numerical columns use the median; categorical columns use
# the mode, or "Unknown" if no mode is available.
for column in df.columns:
    if df[column].isnull().sum() > 0:

        if pd.api.types.is_numeric_dtype(df[column]):
            median_value = df[column].median()

            if pd.notna(median_value):
                df[column] = df[column].fillna(median_value)

        else:
            mode_values = df[column].mode(dropna=True)

            if not mode_values.empty:
                df[column] = df[column].fillna(mode_values.iloc[0])
            else:
                df[column] = df[column].fillna("Unknown")

print("Missing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("Total remaining missing values:", df.isnull().sum().sum())

# ------------------------------------------------------------
# 5. Inspect and convert data types if necessary
# ------------------------------------------------------------

print("\n4. DATA TYPE ANALYSIS")
print("-" * 40)

print("Data types before conversion:")
print(df.dtypes)

# Convert service availability columns to categorical data.
categorical_columns = [
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu"
]

for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].astype("category")

# Convert known numeric columns only when they exist.
numeric_columns = [
    "Country Code",
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes",
    "Longitude",
    "Latitude"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nData types after conversion:")
print(df.dtypes)

# ------------------------------------------------------------
# 6. Analyze Aggregate rating distribution
# ------------------------------------------------------------

print("\n5. AGGREGATE RATING DISTRIBUTION")
print("-" * 40)

if "Aggregate rating" not in df.columns:
    raise KeyError("The dataset does not contain 'Aggregate rating'.")

print("\nRating descriptive statistics:")
print(df["Aggregate rating"].describe())

print("\nRating frequencies:")
rating_counts = df["Aggregate rating"].value_counts(
    dropna=False
).sort_index()

print(rating_counts)

# ------------------------------------------------------------
# 7. Identify potential class imbalance
# ------------------------------------------------------------

print("\n6. RATING GROUP ANALYSIS")
print("-" * 40)

# In this dataset, a rating of 0 means Not rated.
# Separate unrated restaurants from positively rated restaurants.
rating_group = pd.Series(
    "Rated",
    index=df.index,
    dtype="object"
)

rating_group[df["Aggregate rating"] == 0] = "Not rated"
rating_group[df["Aggregate rating"].isna()] = "Missing rating"

group_counts = rating_group.value_counts()
group_percentages = rating_group.value_counts(normalize=True) * 100

rating_summary = pd.DataFrame({
    "Count": group_counts,
    "Percentage": group_percentages.round(2)
})

print(rating_summary)

print(
    "\nNote: A rating of 0.0 represents 'Not rated' "
    "in this dataset, not necessarily a poor rating."
)

# ------------------------------------------------------------
# 8. Visualize rating distribution
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")

# Chart 1: Distribution of aggregate ratings
plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Aggregate rating",
    bins=20,
    kde=False
)

plt.title("Distribution of Restaurant Aggregate Ratings")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "aggregate_rating_distribution.png",
    dpi=300
)

plt.show()

# Chart 2: Rated versus not-rated restaurants
plt.figure(figsize=(8, 5))

sns.countplot(
    x=rating_group,
    order=["Not rated", "Rated", "Missing rating"]
)

plt.title("Rated vs Not-Rated Restaurants")
plt.xlabel("Rating Group")
plt.ylabel("Number of Restaurants")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rating_group_distribution.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 9. Save cleaned dataset and summary reports
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "dataset_preprocessed.csv",
    index=False
)

missing_values.to_csv(
    OUTPUT_DIR / "missing_values_summary.csv",
    header=["Missing Values"]
)

rating_summary.to_csv(
    OUTPUT_DIR / "rating_group_summary.csv"
)

rating_counts.to_csv(
    OUTPUT_DIR / "rating_frequency.csv",
    header=["Count"]
)

# Save a simple text report.
with open(
    OUTPUT_DIR / "task1_report.txt",
    "w",
    encoding="utf-8"
) as report:

    report.write("COGNIFYZ DATA SCIENCE INTERNSHIP\n")
    report.write("LEVEL 1 - TASK 1\n")
    report.write("Data Exploration and Preprocessing\n\n")

    report.write(f"Total rows: {df.shape[0]}\n")
    report.write(f"Total columns: {df.shape[1]}\n")
    report.write(
        f"Missing values after cleaning: "
        f"{df.isnull().sum().sum()}\n\n"
    )

    report.write("Rating group summary:\n")
    report.write(rating_summary.to_string())
    report.write("\n\nRating descriptive statistics:\n")
    report.write(
        df["Aggregate rating"].describe().to_string()
    )

print("\n7. OUTPUT FILES")
print("-" * 40)
print("Preprocessed dataset saved.")
print("Missing-value summary saved.")
print("Rating summaries saved.")
print("Charts saved in the outputs folder.")
print("Task 1 report saved.")

print("\nTask 1 analysis completed successfully!")
