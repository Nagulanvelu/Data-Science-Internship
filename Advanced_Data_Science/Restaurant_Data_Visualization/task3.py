
"""
Cognifyz Data Science Internship
Level 3 - Task 3: Data Visualization

Visualizations:
1. Restaurant rating distribution
2. Average ratings across cuisines
3. Average ratings across cities
4. Relationship between votes, cost, and ratings
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# 1. Setup
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

required_columns = [
    "Aggregate rating",
    "Cuisines",
    "City",
    "Votes",
    "Average Cost for two",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Convert numeric columns safely
for col in ["Aggregate rating", "Votes", "Average Cost for two"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Cuisines"] = df["Cuisines"].fillna("Unknown").astype(str)
df["City"] = df["City"].fillna("Unknown").astype(str)

# Exclude unrated restaurants from rating comparisons
rated = df[df["Aggregate rating"] > 0].copy()

sns.set_theme(style="whitegrid")

print("=" * 60)
print("COGNIFYZ INTERNSHIP - LEVEL 3, TASK 3")
print("DATA VISUALIZATION")
print("=" * 60)
print("Dataset shape:", df.shape)
print("Rated restaurants:", len(rated))


# 2. Restaurant rating distribution
plt.figure(figsize=(9, 6))
plt.hist(
    rated["Aggregate rating"].dropna(),
    bins=20,
    edgecolor="black",
)
plt.title("Distribution of Restaurant Aggregate Ratings")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rating_distribution.png", dpi=300)
plt.close()


# 3. Average rating by cuisine
# Split combined cuisine labels into individual cuisine names
cuisine_data = rated[["Cuisines", "Aggregate rating"]].copy()
cuisine_data["Cuisine"] = cuisine_data["Cuisines"].str.split(",")
cuisine_data = cuisine_data.explode("Cuisine")
cuisine_data["Cuisine"] = cuisine_data["Cuisine"].str.strip()
cuisine_data = cuisine_data[cuisine_data["Cuisine"] != ""]

cuisine_summary = (
    cuisine_data.groupby("Cuisine")
    .agg(
        Restaurant_Count=("Aggregate rating", "size"),
        Average_Rating=("Aggregate rating", "mean"),
    )
    .reset_index()
)

# Require at least 10 restaurants for a more stable comparison
cuisine_summary = cuisine_summary[
    cuisine_summary["Restaurant_Count"] >= 10
]

top_cuisines = cuisine_summary.nlargest(10, "Average_Rating")
top_cuisines.to_csv(
    OUTPUT_DIR / "average_rating_by_cuisine.csv",
    index=False,
)

plt.figure(figsize=(11, 6))
plot_cuisines = top_cuisines.sort_values("Average_Rating")
plt.barh(
    plot_cuisines["Cuisine"],
    plot_cuisines["Average_Rating"],
)
plt.title("Top 10 Cuisines by Average Restaurant Rating")
plt.xlabel("Average Rating")
plt.ylabel("Cuisine")
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "average_rating_by_cuisine.png", dpi=300)
plt.close()


# 4. Average rating by city
city_summary = (
    rated.groupby("City")
    .agg(
        Restaurant_Count=("Aggregate rating", "size"),
        Average_Rating=("Aggregate rating", "mean"),
    )
    .reset_index()
)

city_summary.to_csv(
    OUTPUT_DIR / "average_rating_by_city.csv",
    index=False,
)

# Compare cities with at least 10 rated restaurants
eligible_cities = city_summary[
    city_summary["Restaurant_Count"] >= 10
]

top_cities = eligible_cities.nlargest(10, "Average_Rating")
top_cities.to_csv(
    OUTPUT_DIR / "top_10_cities_by_average_rating.csv",
    index=False,
)

plt.figure(figsize=(11, 6))
plot_cities = top_cities.sort_values("Average_Rating")
plt.barh(
    plot_cities["City"],
    plot_cities["Average_Rating"],
)
plt.title("Top 10 Cities by Average Restaurant Rating")
plt.xlabel("Average Rating")
plt.ylabel("City")
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "average_rating_by_city.png", dpi=300)
plt.close()


# 5. Votes versus aggregate rating
vote_data = rated.dropna(
    subset=["Votes", "Aggregate rating"]
).copy()

plt.figure(figsize=(9, 6))
plt.scatter(
    vote_data["Votes"],
    vote_data["Aggregate rating"],
    alpha=0.35,
)
plt.title("Customer Votes vs Restaurant Rating")
plt.xlabel("Number of Votes")
plt.ylabel("Aggregate Rating")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "votes_vs_rating.png", dpi=300)
plt.close()


# 6. Average cost versus aggregate rating
cost_data = rated.dropna(
    subset=["Average Cost for two", "Aggregate rating"]
).copy()

plt.figure(figsize=(9, 6))
plt.scatter(
    cost_data["Average Cost for two"],
    cost_data["Aggregate rating"],
    alpha=0.35,
)
plt.title("Average Cost for Two vs Restaurant Rating")
plt.xlabel("Average Cost for Two")
plt.ylabel("Aggregate Rating")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cost_vs_rating.png", dpi=300)
plt.close()


# 7. Correlation heatmap for numeric features
numeric_columns = [
    "Aggregate rating",
    "Votes",
    "Average Cost for two",
    "Price range",
]

available_numeric = [
    col for col in numeric_columns if col in df.columns
]

for col in available_numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")

correlation = df[available_numeric].corr()

correlation.to_csv(OUTPUT_DIR / "numeric_feature_correlations.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
)
plt.title("Correlation Between Numeric Restaurant Features")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "numeric_correlation_heatmap.png", dpi=300)
plt.close()


# 8. Write report
report_path = OUTPUT_DIR / "task3_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("COGNIFYZ DATA SCIENCE INTERNSHIP\n")
    report.write("LEVEL 3 - TASK 3: DATA VISUALIZATION\n\n")
    report.write(f"Total dataset records: {len(df)}\n")
    report.write(f"Rated restaurant records: {len(rated)}\n\n")

    report.write("TOP CUISINES BY AVERAGE RATING\n")
    report.write(top_cuisines.to_string(index=False))
    report.write("\n\nTOP CITIES BY AVERAGE RATING\n")
    report.write(top_cities.to_string(index=False))
    report.write("\n\nNUMERIC FEATURE CORRELATIONS\n")
    report.write(correlation.to_string())
    report.write(
        "\n\nInterpretation notes:\n"
        "- Only restaurants with ratings above zero were used for "
        "rating comparisons.\n"
        "- City comparisons include cities with at least 10 rated "
        "restaurants.\n"
        "- Cuisine comparisons include cuisines with at least 10 "
        "restaurant entries.\n"
        "- Correlation indicates association, not causation.\n"
        "- A restaurant listing multiple cuisines contributes to "
        "each cuisine's summary.\n"
    )

print("\nTop cuisines by average rating:")
print(top_cuisines.to_string(index=False))

print("\nTop cities by average rating:")
print(top_cities.to_string(index=False))

print("\nOutputs saved to:", OUTPUT_DIR.resolve())
print("Task 3 completed successfully.")
