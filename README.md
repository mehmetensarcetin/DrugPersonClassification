# Patient Clustering for Drug Prescription Patterns - (cluster_patients.py)

## Overview

This project clusters patients based on various features like age, sex, blood pressure (BP), cholesterol, and Na-to-K ratio using the K-Means clustering algorithm. The goal is to identify patterns in drug prescriptions that could help in medical decision-making.

## Features

- **Data Preprocessing**: Categorical data is dynamically encoded into numeric values based on data types.
- **Standardization**: The dataset is standardized for better performance with the K-Means algorithm.
- **Clustering**: The K-Means algorithm is used to group patients into clusters.
- **Elbow Method**: Used to determine the optimal number of clusters. Implemented as an independent function for reusability.
- **Cluster Analysis**: Analyzes and summarizes the characteristics of each cluster.


# Drug Prescription Prediction - (drug_prescription_prediction.py)

This code contains a Python application that predicts the drug prescribed to a patient based on their features such as age, sex, blood pressure (BP), cholesterol levels, and Na_to_K ratio using machine learning models.

## Features

- **Data Preprocessing**: The application preprocesses the data by encoding categorical features.
- **Model Training**: Three machine learning models are trained:
  - Decision Tree
  - Random Forest
  - Logistic Regression
- **Model Evaluation**: The models are evaluated based on accuracy and classification reports.
