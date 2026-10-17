# Cognifyz Data Science Internship Project

## Project Overview

This project involves analyzing restaurant data using Python to extract insights about restaurant ratings, cuisine preferences, geographic distribution, and performance. It includes data preprocessing, descriptive analytics, geospatial analysis, predictive modeling, and data visualization.

The project consists of six analytical modules organized into two main sections.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Git and GitHub

## Project Structure

```text
Cognifyz-Data-Science-Internship/
│
├── Exploratory_Data_Analysis/
│   ├── Data_Exploration_Preprocessing/
│   │   ├── data_quality_analysis.py
│   │   ├── Dataset.csv
│   │   └── outputs/
│   ├── Descriptive_Analytics/
│   │   ├── descriptive_analytics.py
│   │   ├── Dataset.csv
│   │   └── outputs/
│   └── Geospatial_Restaurant_Analysis/
│       ├── geospatial_analysis.py
│       ├── Dataset.csv
│       └── outputs/
│
├── Advanced_Data_Science/
│   ├── Restaurant_Rating_Prediction/
│   │   ├── restaurant_rating_model.py
│   │   ├── Dataset.csv
│   │   └── outputs/
│   ├── Cuisine_Preference_Analytics/
│   │   ├── cuisine_preference_analysis.py
│   │   ├── Dataset.csv
│   │   └── outputs/
│   └── Restaurant_Data_Visualization/
│       ├── restaurant_performance_visualization.py
│       ├── Dataset.csv
│       └── outputs/
│
└── README.md
```

## Module 1: Data Exploration and Preprocessing

- Examined dataset structure, column types, and missing values.
- Preprocessed restaurant records and analyzed rating distributions.
- Grouped restaurants by rating categories.
- Generated cleaned datasets, summary reports, and charts.

## Module 2: Descriptive Analytics

- Calculated descriptive statistics for numerical variables.
- Analyzed restaurant counts by city and country.
- Identified the most frequently represented cuisines.
- Generated summary tables and visualizations.

## Module 3: Geospatial Restaurant Analysis

- Examined restaurant latitude and longitude coordinates.
- Analyzed restaurant distributions across cities.
- Investigated relationships between geographic coordinates and restaurant ratings.
- Generated location maps and analytical reports.

## Module 4: Restaurant Rating Prediction

- Developed regression models to predict aggregate restaurant ratings.
- Compared Linear Regression, Decision Tree, and Random Forest models.
- Evaluated models using MAE, MSE, RMSE, and R².

**Best-performing model:** Random Forest Regression

**Test R² score:** 0.6397

## Module 5: Cuisine Preference Analytics

- Analyzed cuisine-level restaurant ratings and customer votes.
- Identified popular cuisines based on total recorded votes.
- Compared average ratings across cuisine categories.
- Generated cuisine rankings, summary tables, and charts.

**Highest total votes:** North Indian — 595,194

**Highest average-rated cuisine:** Kebab — 4.38/5, based on 10 restaurant entries.

## Module 6: Restaurant Data Visualization

- Visualized aggregate rating distributions.
- Compared average ratings across cuisines and cities.
- Examined relationships between votes, restaurant costs, and ratings.
- Generated correlation heatmaps and analytical charts.

**Highest average-rated city in the analysis:** London — 4.535/5.

## Installation and Setup

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Execution

Navigate to the relevant module directory and run its Python script:

```bash
python actual_script_name.py
```

Run each script from its own directory, where the corresponding `Dataset.csv` file is located. Generated charts, CSV summaries, and reports are saved in the module's `outputs` directory.

## Evaluation Metrics

The restaurant rating prediction module uses the following metrics:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

## Notes

- Unrated restaurant records are excluded from rating comparisons where specified.
- Cuisine and city rankings depend on the dataset and analysis filters.
- Correlation indicates association and does not establish causation.
- Model performance reflects the dataset and train/test split used.

## Author

**Nagulan V**

B.Tech Artificial Intelligence  
St. Joseph's Institute of Technology

## Acknowledgment

This project was completed as part of the Cognifyz Data Science Internship.
