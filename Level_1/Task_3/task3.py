
import pandas as pd
import matplotlib.pyplot as plt
import folium
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the dataset
df = pd.read_csv(BASE_DIR / "Dataset.csv")

print("=" * 60)
print("COGNIFYZ DATA SCIENCE INTERNSHIP")
print("LEVEL 1 - TASK 3: GEOSPATIAL ANALYSIS")
print("=" * 60)

# 1. Prepare valid geographic coordinates
geo_df = df.dropna(
    subset=["Latitude", "Longitude"]
).copy()

geo_df = geo_df[
    geo_df["Latitude"].between(-90, 90)
    & geo_df["Longitude"].between(-180, 180)
]

print("\n1. GEOGRAPHIC DATA")
print("Total restaurants:", len(df))
print("Restaurants with valid coordinates:", len(geo_df))
print("Restaurants excluded:", len(df) - len(geo_df))

# 2. Create an interactive restaurant map
# Center the map on the average coordinates
center_lat = geo_df["Latitude"].mean()
center_lon = geo_df["Longitude"].mean()

restaurant_map = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5
)

# Limit map markers to avoid an excessively large HTML file
map_df = geo_df.sample(
    n=min(3000, len(geo_df)),
    random_state=42
)

for _, row in map_df.iterrows():
    rating = row["Aggregate rating"]

    if rating == 0:
        marker_color = "gray"
    elif rating < 2.5:
        marker_color = "red"
    elif rating < 3.5:
        marker_color = "orange"
    elif rating < 4.0:
        marker_color = "blue"
    else:
        marker_color = "green"

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=3,
        color=marker_color,
        fill=True,
        fill_opacity=0.6,
        popup=(
            f"Restaurant: {row['Restaurant Name']}<br>"
            f"City: {row['City']}<br>"
            f"Rating: {rating}"
        )
    ).add_to(restaurant_map)

map_path = OUTPUT_DIR / "restaurant_locations_map.html"
restaurant_map.save(str(map_path))

print("\nInteractive map saved:", map_path)
print("Map markers displayed:", len(map_df))

# 3. Analyze restaurant distribution by city
city_counts = (
    geo_df["City"]
    .fillna("Unknown")
    .value_counts()
)

print("\n2. TOP 10 CITIES BY RESTAURANT COUNT")
print(city_counts.head(10))

city_counts.rename_axis("City").reset_index(
    name="Restaurant Count"
).to_csv(
    OUTPUT_DIR / "restaurant_counts_by_city.csv",
    index=False
)

top_cities = city_counts.head(10).sort_values()

plt.figure(figsize=(10, 6))
top_cities.plot(kind="barh")
plt.title("Top 10 Cities by Restaurant Count")
plt.xlabel("Number of Restaurants")
plt.ylabel("City")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "restaurant_distribution_by_city.png",
    dpi=300
)
plt.close()

# 4. Examine ratings by city
city_rating = (
    geo_df.groupby("City")
    .agg(
        Restaurant_Count=("Restaurant ID", "count"),
        Average_Rating=("Aggregate rating", "mean")
    )
    .sort_values("Restaurant_Count", ascending=False)
)

city_rating.to_csv(
    OUTPUT_DIR / "city_rating_analysis.csv"
)

print("\n3. CITY-WISE RATINGS")
print(city_rating.head(10).round(2))

# 5. Correlation between coordinates and ratings
rating_df = geo_df[
    geo_df["Aggregate rating"] > 0
].copy()

correlations = rating_df[
    ["Latitude", "Longitude", "Aggregate rating"]
].corr()

print("\n4. COORDINATE AND RATING CORRELATIONS")
print(correlations.round(4))

correlations.to_csv(
    OUTPUT_DIR / "location_rating_correlations.csv",
    index_label="Variable"
)

# Visualize restaurant coordinates, colored by rating
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    rating_df["Longitude"],
    rating_df["Latitude"],
    c=rating_df["Aggregate rating"],
    cmap="viridis",
    s=12,
    alpha=0.6
)

plt.colorbar(scatter, label="Aggregate Rating")
plt.title("Restaurant Locations Colored by Rating")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "restaurant_locations_by_rating.png",
    dpi=300
)
plt.close()

# 6. Save a report
with open(
    OUTPUT_DIR / "task3_report.txt",
    "w",
    encoding="utf-8"
) as report:
    report.write("COGNIFYZ DATA SCIENCE INTERNSHIP\n")
    report.write("LEVEL 1 - TASK 3: GEOSPATIAL ANALYSIS\n\n")
    report.write(f"Total restaurants: {len(df)}\n")
    report.write(
        f"Restaurants with valid coordinates: {len(geo_df)}\n"
    )
    report.write(
        f"Restaurants mapped: {len(map_df)}\n\n"
    )
    report.write("Top 10 cities by restaurant count:\n")
    report.write(city_counts.head(10).to_string())
    report.write("\n\nCity-wise ratings:\n")
    report.write(city_rating.head(10).round(2).to_string())
    report.write("\n\nCoordinate-rating correlations:\n")
    report.write(correlations.round(4).to_string())
    report.write(
        "\n\nNote: Correlation does not establish causation. "
        "Only restaurants with positive ratings were used "
        "for the coordinate-rating correlation calculation.\n"
    )

print("\nTask 3 analysis completed successfully!")
print("All results saved in:", OUTPUT_DIR)
