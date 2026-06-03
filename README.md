# Salary Prediction Using Simple Linear Regression

## Project Overview

This project demonstrates how to build a **Simple Linear Regression** model to predict employee salaries based on their years of experience. The dataset is obtained from Kaggle and contains salary information for employees with varying levels of experience.

## Dataset

**Source:** Kaggle Salary Dataset

The dataset contains the following columns:

| Column          | Description                                 |
| --------------- | ------------------------------------------- |
| YearsExperience | Number of years of professional experience  |
| Salary          | Employee salary                             |
| Unnamed: 0      | Index column (removed during preprocessing) |

## Libraries Used

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

## Steps Performed

### 1. Download Dataset

The dataset was downloaded using `kagglehub`.

```python
import kagglehub

path = kagglehub.dataset_download(
    "abhishek14398/salary-dataset-simple-linear-regression"
)
```

### 2. Load Dataset

```python
data = pd.read_csv(f"{path}/Salary_dataset.csv")
```

### 3. Data Exploration

* Displayed first few rows using `head()`
* Checked dataset information using `info()`
* Calculated correlation between features

```python
data.head()
data.info()
data.corr()
```

### 4. Data Cleaning

#### Handle Missing Values

```python
data.dropna(inplace=True)
```

#### Remove Duplicate Records

```python
data.drop_duplicates(inplace=True)
```

#### Drop Unnecessary Column

```python
data.drop('Unnamed: 0', axis=1, inplace=True)
```

### 5. Feature Selection

#### Independent Variable (X)

```python
x = data.drop('Salary', axis=1)
```

#### Dependent Variable (y)

```python
y = data['Salary']
```

### 6. Split Dataset

The dataset was split into training and testing sets.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

* Training Data: 80%
* Testing Data: 20%

## Dataset Summary

* Total Records: 30
* Features Used: 1 (`YearsExperience`)
* Target Variable: `Salary`
* Missing Values: None
* Duplicate Values: None

## Correlation Analysis

| Feature         | Correlation with Salary |
| --------------- | ----------------------- |
| YearsExperience | 0.978                   |
| Unnamed: 0      | 0.961                   |

The strong positive correlation indicates that salary tends to increase with years of experience.

## Next Steps

After splitting the data, the following steps can be performed:

1. Train a Simple Linear Regression model.
2. Predict salaries on the test set.
3. Evaluate model performance using:

   * Mean Absolute Error (MAE)
   * Mean Squared Error (MSE)
   * Root Mean Squared Error (RMSE)
   * R² Score
4. Visualize the regression line.

Example:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
```

