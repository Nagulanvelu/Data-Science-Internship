
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Create output folder beside this Python file
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(Path(__file__).resolve().parent / "Dataset.csv")

print("=" * 60)
print("COGNIFYZ DATA SCIENCE INTERNSHIP")
print("LEVEL 1 - TASK 2: DESCRIPTIVE ANALYSIS")
print("=" * 60)

# 1. Basic dataset information
print("\n1. DATASET INFORMATION")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn names:")
print(df.columns.tolist())

# 2. Statistical measures for numerical columns
print("\n2. NUMERICAL DESCRIPTIVE STATISTICS")
numerical_stats = df.describe().T
print(numerical_stats.round(2))

numerical_stats.to_csv(
    OUTPUT_DIR / "numerical_statistics.csv"
)

# 3. Explore categorical variables
print("\n3. CATEGORICAL VARIABLE ANALYSIS")

categorical_columns = ["Country Code", "City", "Cuisines"]

for column in categorical_columns:
    print(f"\nTop 10 values in {column}:")
    print(df[column].fillna("Unknown").value_counts().head(10))

# Save frequency tables
for column in categorical_columns:
    counts = (
        df[column]
        .fillna("Unknown")
        .value_counts()
        .rename_axis(column)
        .reset_index(name="Restaurant Count")
    )

    counts.to_csv(
        OUTPUT_DIR / f"{column.lower().replace(' ', '_')}_counts.csv",
        index=False
    )

# 4. Top cuisines
print("\n4. TOP 10 CUISINES")
top_cuisines = (
    df["Cuisines"]
    .fillna("Unknown")
    .value_counts()
    .head(10)
)
print(top_cuisines)

top_cuisines.to_csv(
    OUTPUT_DIR / "top_10_cuisines.csv",
    header=["Restaurant Count"]
)

plt.figure(figsize=(10, 6))
top_cuisines.sort_values().plot(kind="barh")
plt.title("Top 10 Most Common Cuisines")
plt.xlabel("Number of Restaurants")
plt.ylabel("Cuisine")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "top_10_cuisines.png",
    dpi=300
)
plt.close()

# 5. Top cities
print("\n5. TOP 10 CITIES BY RESTAURANT COUNT")
top_cities = df["City"].fillna("Unknown").value_counts().head(10)
print(top_cities)

top_cities.to_csv(
    OUTPUT_DIR / "top_10_cities.csv",
    header=["Restaurant Count"]
)

plt.figure(figsize=(10, 6))
top_cities.sort_values().plot(kind="barh")
plt.title("Top 10 Cities by Number of Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("City")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "top_10_cities.png",
    dpi=300
)
plt.close()

# 6. Save a text report
with open(OUTPUT_DIR / "task2_report.txt", "w", encoding="utf-8") as report:
    report.write("COGNIFYZ DATA SCIENCE INTERNSHIP\n")
    report.write("LEVEL 1 - TASK 2: DESCRIPTIVE ANALYSIS\n\n")
    report.write(f"Total rows: {df.shape[0]}\n")
    report.write(f"Total columns: {df.shape[1]}\n\n")
    report.write("Numerical descriptive statistics:\n")
    report.write(numerical_stats.round(2).to_string())
    report.write("\n\nTop 10 cuisines:\n")
    report.write(top_cuisines.to_string())
    report.write("\n\nTop 10 cities:\n")
    report.write(top_cities.to_string())
    report.write("\n")

print("\nTask 2 analysis completed successfully!")
print("All results saved in:", OUTPUT_DIR)
