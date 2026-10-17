
"""
Cognifyz Data Science Internship
Level 3 - Task 2: Customer Preference Analysis

Requirements:
1. Analyze cuisine type versus restaurant ratings.
2. Find popular cuisines using total customer votes.
3. Identify cuisines with higher average ratings.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# 1. Paths and dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

required_columns = [
    "Cuisines",
    "Aggregate rating",
    "Votes",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["Aggregate rating"] = pd.to_numeric(
    df["Aggregate rating"], errors="coerce"
)
df["Votes"] = pd.to_numeric(
    df["Votes"], errors="coerce"
)

# Exclude missing cuisine names and unrated restaurants
df = df.dropna(
    subset=["Cuisines", "Aggregate rating", "Votes"]
).copy()

df["Cuisines"] = df["Cuisines"].astype(str).str.strip()
df = df[df["Cuisines"] != ""]
rated_df = df[df["Aggregate rating"] > 0].copy()

print("=" * 65)
print("COGNIFYZ INTERNSHIP - LEVEL 3, TASK 2")
print("CUSTOMER PREFERENCE ANALYSIS")
print("=" * 65)

print("\nDataset shape:", df.shape)
print("Rated restaurant records:", len(rated_df))


# --------------------------------------------------
# 2. Cuisine-level analysis
# --------------------------------------------------

# Dataset entries can contain multiple cuisines, such as
# "North Indian, Chinese". Split these into individual
# cuisines so each cuisine can be analyzed separately.

cuisine_rows = rated_df[
    ["Cuisines", "Aggregate rating", "Votes"]
].copy()

cuisine_rows["Cuisine"] = cuisine_rows["Cuisines"].str.split(",")

cuisine_rows = cuisine_rows.explode("Cuisine")
cuisine_rows["Cuisine"] = cuisine_rows["Cuisine"].str.strip()

cuisine_rows = cuisine_rows[
    cuisine_rows["Cuisine"] != ""
].copy()

cuisine_summary = (
    cuisine_rows.groupby("Cuisine")
    .agg(
        Restaurant_Count=("Aggregate rating", "size"),
        Average_Rating=("Aggregate rating", "mean"),
        Median_Rating=("Aggregate rating", "median"),
        Total_Votes=("Votes", "sum"),
        Average_Votes=("Votes", "mean"),
    )
    .reset_index()
)

cuisine_summary = cuisine_summary.sort_values(
    "Restaurant_Count", ascending=False
)

cuisine_summary.to_csv(
    OUTPUT_DIR / "cuisine_rating_summary.csv",
    index=False,
)

print("\nNumber of distinct cuisines:", len(cuisine_summary))


# --------------------------------------------------
# 3. Most popular cuisines by total customer votes
# --------------------------------------------------

top_votes = cuisine_summary.sort_values(
    "Total_Votes", ascending=False
).head(10)

top_votes.to_csv(
    OUTPUT_DIR / "top_10_cuisines_by_votes.csv",
    index=False,
)

print("\nTop 10 cuisines by total votes:")
print(
    top_votes[
        ["Cuisine", "Restaurant_Count", "Total_Votes", "Average_Rating"]
    ].to_string(index=False)
)

plt.figure(figsize=(11, 6))
plt.barh(
    top_votes["Cuisine"][::-1],
    top_votes["Total_Votes"][::-1],
)
plt.title("Top 10 Cuisines by Total Customer Votes")
plt.xlabel("Total Votes")
plt.ylabel("Cuisine")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "top_10_cuisines_by_votes.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# 4. Cuisines with the highest average ratings
# --------------------------------------------------

# Use a minimum restaurant count to avoid ranking a cuisine
# based on only one or two restaurants.
minimum_restaurants = 10

eligible_cuisines = cuisine_summary[
    cuisine_summary["Restaurant_Count"] >= minimum_restaurants
].copy()

top_rated = eligible_cuisines.sort_values(
    "Average_Rating", ascending=False
).head(10)

top_rated.to_csv(
    OUTPUT_DIR / "top_10_cuisines_by_rating.csv",
    index=False,
)

print(
    f"\nTop 10 rated cuisines "
    f"(minimum {minimum_restaurants} restaurants):"
)
print(
    top_rated[
        ["Cuisine", "Restaurant_Count", "Average_Rating", "Total_Votes"]
    ].to_string(index=False)
)

plt.figure(figsize=(11, 6))
plt.barh(
    top_rated["Cuisine"][::-1],
    top_rated["Average_Rating"][::-1],
)
plt.title("Top 10 Cuisines by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Cuisine")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "top_10_cuisines_by_rating.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# 5. Cuisine rating distribution
# --------------------------------------------------

plt.figure(figsize=(9, 6))
plt.hist(
    rated_df["Aggregate rating"],
    bins=20,
    edgecolor="black",
)
plt.title("Distribution of Restaurant Ratings")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "restaurant_rating_distribution.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# 6. Relationship between cuisine and rating
# --------------------------------------------------

# Scatter plot: cuisines with enough restaurants are shown
# against their average ratings.

plot_data = eligible_cuisines.sort_values(
    "Average_Rating", ascending=False
).head(20)

plt.figure(figsize=(12, 7))
plt.scatter(
    plot_data["Restaurant_Count"],
    plot_data["Average_Rating"],
    alpha=0.8,
)
for _, row in plot_data.iterrows():
    plt.annotate(
        row["Cuisine"],
        (
            row["Restaurant_Count"],
            row["Average_Rating"],
        ),
        fontsize=7,
        alpha=0.8,
    )

plt.title("Cuisine Popularity vs Average Rating")
plt.xlabel("Number of Restaurants Serving Cuisine")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "cuisine_count_vs_rating.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# 7. Save a text report
# --------------------------------------------------

report_path = OUTPUT_DIR / "task2_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("COGNIFYZ DATA SCIENCE INTERNSHIP\n")
    report.write("LEVEL 3 - TASK 2: CUSTOMER PREFERENCE ANALYSIS\n\n")

    report.write(f"Rated restaurant records: {len(rated_df)}\n")
    report.write(f"Distinct individual cuisines: {len(cuisine_summary)}\n")
    report.write(
        "Cuisine entries containing multiple cuisine names were "
        "split so each individual cuisine could be analyzed.\n"
    )
    report.write(
        f"Cuisines ranked by rating must have at least "
        f"{minimum_restaurants} restaurants.\n\n"
    )

    report.write("TOP 10 CUISINES BY TOTAL VOTES\n")
    report.write(
        top_votes[
            ["Cuisine", "Restaurant_Count", "Total_Votes", "Average_Rating"]
        ].to_string(index=False)
    )
    report.write("\n\n")

    report.write("TOP 10 CUISINES BY AVERAGE RATING\n")
    report.write(
        top_rated[
            ["Cuisine", "Restaurant_Count", "Average_Rating", "Total_Votes"]
        ].to_string(index=False)
    )
    report.write("\n\n")
    report.write(
        "Interpretation: total votes measure the combined number of "
        "votes recorded for restaurants serving a cuisine. They do "
        "not necessarily measure unique customers. Average ratings "
        "are descriptive associations and do not establish causation.\n"
    )

print("\nOutputs saved to:", OUTPUT_DIR.resolve())
print("Task 2 analysis completed successfully.")
