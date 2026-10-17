# Cognifyz Data Science Internship 🚀

A hands-on Data Science internship project focused on data preprocessing, exploratory analysis, predictive modeling, customer preference analysis, and data visualization using Python.

## 📌 Project Overview

This repository contains completed tasks from **Level 1 and Level 3** of the Cognifyz Data Science Internship.

### 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter/Python scripts
- Git and GitHub

## 📂 Project Structure

```text

Cognifyz-Data-Science-Internship/
├── Exploratory_Data_Analysis/
│   ├── Data_Exploration_Preprocessing/
│   │   ├── task1.py
│   │   ├── Dataset.csv
│   │   └── outputs/
│   ├── Descriptive_Analytics/
│   │   ├── task2.py
│   │   ├── Dataset.csv
│   │   └── outputs/
│   └── Geospatial_Restaurant_Analysis/
│       ├── task3.py
│       ├── Dataset.csv
│       └── outputs/
│
├── Advanced_Data_Science/
│   ├── Restaurant_Rating_Prediction/
│   │   ├── task1.py
│   │   ├── Dataset.csv
│   │   └── outputs/
│   ├── Cuisine_Preference_Analytics/
│   │   ├── task2.py
│   │   ├── Dataset.csv
│   │   └── outputs/
│   └── Restaurant_Data_Visualization/
│       ├── task3.py
│       ├── Dataset.csv
│       └── outputs/
│
└── README.md


```

## 📊 Level 1 — Exploratory Data Analysis

### Task 1: Data Exploration and Preprocessing
- Explored dataset structure and missing values.
- Preprocessed restaurant data.
- Analyzed rating distributions and rating groups.
- Generated summary tables, charts, and a task report.

### Task 2: Descriptive Analysis
- Analyzed numerical statistics.
- Examined city, country, and cuisine distributions.
- Identified the top 10 cities and cuisines.
- Generated CSV summaries and visualizations.

### Task 3: Geospatial Analysis
- Analyzed restaurant locations using latitude and longitude.
- Examined restaurant distributions across cities.
- Investigated relationships between location coordinates and ratings.
- Generated a restaurant location map and supporting reports.

## 🤖 Level 3 — Advanced Analysis

### Task 1: Predictive Modeling
- Built and compared Linear Regression, Decision Tree, and Random Forest models.
- Predicted restaurant aggregate ratings.
- Evaluated performance using MAE, MSE, RMSE, and R².

**Best model:** Random Forest Regression  
**Test R² score:** 0.6397

### Task 2: Customer Preference Analysis
- Analyzed cuisine preferences using restaurant ratings and votes.
- Identified popular cuisines by total votes.
- Compared average ratings across cuisines.
- Generated ranked CSV files and charts.

**Most-voted cuisine:** North Indian — 595,194 total votes  
**Highest average-rated cuisine:** Kebab — 4.38/5, based on 10 restaurant entries

### Task 3: Data Visualization
- Visualized restaurant rating distributions.
- Compared average ratings across cuisines and cities.
- Examined votes versus ratings and cost versus ratings.
- Generated a numeric correlation heatmap.

**Highest average-rated city in the analysis:** London — 4.535/5

## ⚙️ Installation and Usage

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Cognifyz-Data-Science-Internship
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run a task

Navigate to the required task folder and run its script:

```bash
python task1.py
python task2.py
python task3.py
```

Run each script from its own task folder, where its `Dataset.csv` file is located.

## 📈 Skills Demonstrated

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Statistical summaries
- Data visualization
- Geospatial analysis
- Regression modeling
- Model evaluation
- Customer preference analysis
- Python scripting and project organization

## ⚠️ Notes

- Restaurant rating comparisons exclude unrated records where specified in the analysis.
- Cuisine and city rankings depend on the available dataset and filtering criteria.
- Correlation indicates association and does not establish causation.
- Model performance reflects the dataset and train/test split used in this project.

## 👨‍💻 Author

**Nagulan V**

B.Tech Artificial Intelligence  
St. Joseph's Institute of Technology

---

*Completed as part of the Cognifyz Data Science Internship.*
