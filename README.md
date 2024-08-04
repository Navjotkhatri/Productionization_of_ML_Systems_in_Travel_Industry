# Productionization_of_ML_Systems_in_Travel_Industry
The goal is to leverage datasets to build and deploy machine learning models, serving a dual purpose: enhancing predictive capabilities in travel-related decision-making and mastering the art of MLOps through hands-on application.

# 1. Flight Price Prediction

<p>
    <img src="https://img.shields.io/badge/Model-Linear%20Regression-blue" alt="Linear Regression" />
    <img src="https://img.shields.io/badge/Model-Lasso%20Regression-blue" alt="Lasso Regression" />
    <img src="https://img.shields.io/badge/Model-Ridge%20Regression-blue" alt="Ridge Regression" />
    <img src="https://img.shields.io/badge/Model-Elastic%20Net%20Regression-blue" alt="Elastic Net Regression" />
    <img src="https://img.shields.io/badge/Model-LightGBM-blue" alt="LightGBM" />
    <img src="https://img.shields.io/badge/Model-XGBoost-blue" alt="XGBoost" />
</p>

<p>
    <img src="https://img.shields.io/badge/Skill-Machine%20Learning-green" alt="Machine Learning" />
    <img src="https://img.shields.io/badge/Skill-Feature%20Engineering-yellow" alt="Feature Engineering" />
    <img src="https://img.shields.io/badge/Skill-Model%20Evaluation-red" alt="Model Evaluation" />
</p>

<p>
    <img src="https://img.shields.io/badge/Tool-Python-blue" alt="Python" />
    <img src="https://img.shields.io/badge/Tool-Scikit%20Learn-yellow" alt="Scikit Learn" />
    <img src="https://img.shields.io/badge/Tool-Pandas-blue" alt="Pandas" />
    <img src="https://img.shields.io/badge/Tool-Numpy-green" alt="Numpy" />
    <img src="https://img.shields.io/badge/Tool-Colab%20Notebook-orange" alt="Colab Notebook" />
</p>


## Project Overview

This project aims to predict flight prices based on various features such as distance, flight type, and other relevant variables. We use different machine learning models to analyze and forecast flight prices, providing insights into pricing strategies and customer behavior.

## Project Description

The goal of this project is to develop a predictive model for flight prices using historical flight data. The dataset includes features such as flight distance, time, price, and flight type. Various machine learning techniques were employed to build and evaluate models.

## Data Analysis

1. **Price Distribution:** Analyzed the distribution of flight prices. Most prices fall between $600 and $900.
2. **Flight Types and Agencies:** Evaluated the market share of flight types and agencies. First-class tickets had the highest share, while "Flying Drops" had higher average prices despite a smaller market share.
3. **Distance and Time:** Examined the correlation between flight distance, time, and price. A positive correlation was noted between distance and price.

## Feature Engineering

- Created new features such as flight route, speed, and temporal components (e.g., day of the week, month).
- Applied scaling techniques to normalize data.

## Modeling

1. **Linear Regression:** Established baseline performance with moderate accuracy.
2. **Regularization Models:** Applied Lasso, Ridge, and Elastic Net regularizations. Slight improvements were observed but did not significantly outperform the baseline.
3. **LightGBM:** Achieved significant improvements in MAE (~$23.81) and R² (~0.993), demonstrating high accuracy.
4. **XGBoost:** Provided the best performance with MAE (~$23.35) and R² (~0.993), making it the final model of choice.

## Model Performance

The following table summarizes the performance metrics for each model:

| MODEL                   | MAE   | MSE      | RMSE  | MAPE  | R2    | Accuracy |
|-------------------------|-------|----------|-------|-------|-------|----------|
| Linear Regression      | 250.13| 97548.73 | 312.32| 30.40%| 0.2622| 69.590%  |
| Lasso Regression        | 250.13| 97548.73 | 312.30| 30.40%| 0.2622| 69.590%  |
| Ridge Regression        | 250.13| 97548.77 | 312.30| 30.41%| 0.2625| 69.540%  |
| Elastic Net Regression  | 250.13| 97548.76 | 312.32| 30.41%| 0.2622| 69.590%  |
| LightGBM                | 23.81 | 900.92   | 30.015| 2.76% | 0.9931| 97.293%  |
| XGBoost                 | 23.35 | 899.44   | 29.990| 2.69% | 0.9930| 97.300%  |

## Feature Importance

XGBoost's feature importance analysis identified key variables affecting flight prices, such as distance and total time. This analysis provides insights into which factors are most influential in predicting flight prices.

## Conclusion

The project successfully combined data analysis, feature engineering, and advanced modeling techniques to develop a robust predictive model for flight prices. XGBoost was selected for its superior performance in accuracy and predictive power. The insights gained can help refine pricing strategies and optimize revenue management.


