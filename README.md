# Optimizing-Delivery-Time-Predictions-Using-Machine-Learning-Models

## Project Overview
This project focuses on building machine learning models to predict delivery times for an online food delivery service. By leveraging various features such as geolocation, traffic density, weather conditions, and delivery personnel data, the project aims to improve the accuracy of delivery time predictions and enhance overall operational efficiency.

## Objective
To develop and compare regression models that can accurately forecast delivery times, allowing businesses to streamline their delivery operations and improve customer satisfaction.

## Dataset
The dataset contains various features including:

Geospatial Data: Restaurant and delivery locations (latitude and longitude)
Delivery Personnel Information: Age, ratings, and ID
Order Information: Time ordered, time picked, and delivery time taken
Weather and Traffic Conditions: Weather descriptions and road traffic density
## Key Steps in the Project
### Data Preprocessing:

Handling missing values (imputation with median/mode).
Feature engineering (e.g., calculating delivery distance using geodesic formulas).
Encoding categorical variables and scaling numerical features

.
### Exploratory Data Analysis (EDA):

Visualizing relationships between key features like delivery distance, traffic density, and delivery time.
Distribution analysis of delivery personnel ratings and age.

### Model Training and Evaluation:

Models used: Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
Dataset split into training and testing sets.
Performance evaluation using R-squared score, with further analysis through additional metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
### Model Comparison:

Linear Regression for basic linear relationships.
Decision Tree for capturing non-linear interactions.
Random Forest for robust, ensemble-based predictions.

## Technologies Used
Python for data processing and model implementation.
Pandas and NumPy for data manipulation.
Seaborn and Matplotlib for visualizations.
Scikit-learn for model building and evaluation.

