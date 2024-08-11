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

# 2. Flight Price Prediction REST API

REST API for predicting flight prices. The API is built using Flask and is accessible over the internet using `ngrok`.

## Features

- Predict flight prices based on various input parameters such as departure city, destination city, flight type, agency, date, and more.
- Simple and clean web form for inputting data and receiving predictions.
- Easily accessible via a public URL using `ngrok`.

## Technologies Used

<p>
    <img src="https://img.shields.io/badge/Language-Python-blue" alt="Python" />
    <img src="https://img.shields.io/badge/Language-HTML/CSS-blue" alt="HTML/CSS" />
    <img src="https://img.shields.io/badge/Framework-Flask-green" alt="Flask" />
    <img src="https://img.shields.io/badge/Tool-ngrok-orange" alt="ngrok" />
    <img src="https://img.shields.io/badge/Tool-Pandas-blue" alt="Pandas" />
    <img src="https://img.shields.io/badge/Tool-Numpy-green" alt="Numpy" />
    <img src="https://img.shields.io/badge/Tool-Scikit%20Learn-yellow" alt="Scikit Learn" />
    <img src="https://img.shields.io/badge/Notebook-Google%20Colab-orange" alt="Google Colab" />
</p>

## Form Preview

<p align="center">
    <img src="https://github.com/Navjotkhatri/Productionization_of_ML_Systems_in_Travel_Industry/blob/main/Screenshot%202024-08-10%20160011.png?raw=true" alt="Flight Price Prediction Form" width="600"/>
</p>


# 3. Containerization

In this step, the flight price prediction model was containerized using Docker to ensure seamless deployment and portability across different environments. Packaging the model, its dependencies, and the Flask web application into a Docker container, eliminates the "it works on my machine" problem, ensuring consistent behaviour regardless of where the container is deployed.

## Key Steps:

- Dockerfile Creation: A Dockerfile was crafted to define the environment setup, including the installation of required libraries and the copying of necessary files into the container.

- Environment Configuration: The model and web application dependencies were listed in a requirements.txt file, which was used in the Dockerfile to install all necessary Python packages.

- Building the Docker Image: The Docker image was built using the command docker build -t flight_price_prediction ., which encapsulated the entire application.

- Running the Container: The application was launched in a Docker container using docker run -p 8000:8000 flight_price_prediction, making the Flask web application accessible at http://localhost:8000.

- Portability and Deployment: The Docker image can be deployed on any platform that supports Docker, ensuring the model is easily portable and deployable in different environments, including cloud platforms, 
   local machines, or other servers.

<p align="center">
    <img src="https://github.com/Navjotkhatri/Productionization_of_ML_Systems_in_Travel_Industry/blob/main/Screenshot%202024-08-11%20132155.png?raw=true" alt="Flight Price Prediction Form" width="600"/>
</p>

# 8. Gender Classification Project

<p>
    <img src="https://img.shields.io/badge/Model-Logistic%20Regression-blue" alt="Logistic Regression" />
    <img src="https://img.shields.io/badge/Model-Random%20Forest%20Classifier-blue" alt="Random Forest Classifier" />
    <img src="https://img.shields.io/badge/Model-XGBoost-blue" alt="XGBoost" />
    <img src="https://img.shields.io/badge/Model-KNN-blue" alt="K-Nearest Neighbors" />
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
The goal of this project is to develop a machine learning model to classify gender based on various user features such as company affiliation, age, and other attributes. The project involves data exploration, preprocessing, model training, evaluation, and interpretability analysis.

## Dataset
The dataset contains user information with features including:
- `code`
- `company`
- `name`
- `age`
- `gender`

The dataset is balanced with three gender categories: male, female, and others.

## Exploratory Data Analysis (EDA)
EDA was performed to understand the distribution and relationships of the features:
- Analyzed the distribution of gender, age, and company.
- Visualized the data using bar charts, pie charts, and distribution plots.
- Checked for correlations between features using a heatmap.

## Data Preprocessing
Data preprocessing steps included:
- Encoding categorical variables using one-hot encoding for `gender` and `company`.
- Applying sentence transformers and PCA for the `name` variable.
- Scaling numerical features using StandardScaler.

## Modeling
Several machine learning models were trained and evaluated:

### Logistic Regression
- Achieved 100% accuracy on both training and test datasets.
- Selected for its high performance and interpretability.

### Random Forest Classifier
- Achieved 100% accuracy on both training and test datasets.
- Offers robust performance and easy interpretability.

### XGBoost
- Tuned using GridSearchCV for optimal hyperparameters.
- Provided strong performance with complex datasets.

### K-Nearest Neighbors (KNN)
- Achieved 89% accuracy on the training set and 77% on the test set.
- Considered for its simplicity and reasonable performance.

## Model Evaluation
Models were evaluated using the following metrics:

| Model                   | Test Accuracy | Test Precision | Test Recall |
|-------------------------|---------------|----------------|-------------|
| Logistic Regression     | 1.00          | 1.00           | 1.00        |
| Random Forest Classifier| 1.00          | 1.00           | 1.00        |
| XGBoost                 | 1.00          | 1.00           | 1.00        |
| KNN                     | 0.77          | 0.79           | 0.77        |


## Conclusion
The logistic regression model was selected as the final model due to its perfect accuracy and ease of interpretation. SHAP analysis confirmed the model's transparency, making it suitable for deployment and decision-making.



