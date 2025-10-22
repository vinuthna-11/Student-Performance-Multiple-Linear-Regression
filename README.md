# Student Performance Prediction — Multiple Linear Regression

> Predicting students' final grades using multiple linear regression and exploratory data analysis.

---

## Table of contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Dataset](#dataset)
* [Repository structure](#repository-structure)
* [Installation / Setup](#installation--setup)
* [Usage](#usage)
* [Methodology](#methodology)
* [Model training & evaluation](#model-training--evaluation)
* [Results](#results)
* [How to contribute](#how-to-contribute)
* [License](#license)

---

## Project Overview

This project implements a **Multiple Linear Regression** model to predict student academic performance (final grade) based on a variety of features such as study hours, demographics, prior grades, attendance, and other socio-academic factors. The repository contains data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and visualization scripts.

## Features

* Data cleaning and preprocessing pipelines
* Exploratory Data Analysis with plots and summary statistics
* Feature selection and transformation
* Multiple Linear Regression model implementation (scikit-learn)
* Model evaluation (MAE, MSE, RMSE, R²)
* Jupyter notebooks demonstrating steps end-to-end

## Dataset

**Dataset name:** `Student Performance` (CSV)

**Typical columns (example):**

* `school`, `sex`, `age`, `address`, `famsize`, `Pstatus`
* `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`
* `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`
* `absences`, `G1`, `G2`, `G3` (where `G3` is typically the final grade to predict)

> **Dataset URL:** Add the dataset file in the `data/` folder of this repository and update this README with the exact path or external URL (e.g., a Kaggle link) if you host it elsewhere.

## Repository structure

```
Student-Performance-Multiple-Linear-Regression/
├─ data/
│  └─ student-performance.csv     # dataset (add here)
├─ notebooks/
│  └─ EDA.ipynb
│  └─ model_training.ipynb
├─ src/
│  ├─ data_preprocessing.py
│  ├─ features.py
│  ├─ train_model.py
│  └─ evaluate.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

Adjust filenames if your repository uses different names — the above is a recommended structure.

## Installation / Setup

1. Clone the repo:

```bash
git clone https://github.com/vinuthna-11/Student-Performance-Multiple-Linear-Regression.git
cd Student-Performance-Multiple-Linear-Regression
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
joblib
```

## Usage

1. Place the dataset CSV in `data/student-performance.csv` (or update paths in scripts/notebooks).
2. Run the EDA notebook to inspect data and visualizations:

```bash
jupyter notebook notebooks/EDA.ipynb
```

3. Preprocess data and train the model from script:

```bash
python src/train_model.py --data data/student-performance.csv --output models/linear_regression.joblib
```

4. Evaluate the model:

```bash
python src/evaluate.py --model models/linear_regression.joblib --data data/student-performance.csv
```

## Methodology

1. **Exploratory Data Analysis (EDA)** — inspect distributions, missing values, correlations, and relationships between features and target (`G3`).
2. **Preprocessing** — handle missing values, encode categorical variables (one-hot / label encoding), scale numeric features if needed, and create derived features (e.g., total prior score `G1+G2`).
3. **Feature Selection** — use correlation analysis and/or feature importance to pick a relevant subset for the regression model.
4. **Modeling** — fit a Multiple Linear Regression model (ordinary least squares) using scikit-learn's `LinearRegression`.
5. **Evaluation** — report MAE, MSE, RMSE, and R² on a held-out test set or using cross-validation.

## Model training & evaluation

* Recommended split: 80% train / 20% test or use K-fold cross-validation (e.g., K=5).
* Save the final model using `joblib.dump`.
* Track baseline performance using a simple mean predictor to quantify model improvement.

**Evaluation metrics:**

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R-squared (R²)
