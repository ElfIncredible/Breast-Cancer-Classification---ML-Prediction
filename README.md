# Breast Cancer Classification - ML Prediction
The goal is to predict whether someone has breast cancer or not.  
** Demonstrate a typical workflow for analyzing a dataset, training a machine learning model, evaluating its performance, and making predictions on new data for a binary classification task related to breast cancer detection.
## Table of Content
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Machine Learning Prediction](#machine-learning-prediction)
  - [Install dependencies](#install-dependencies)
  - [Data collection and processing](#data-collection-and-processing)
  - [Separating features and target](#separating-features-and-target)
  - [Split data into train and test data](#split-data-into-train-and-test-data)
  - [Model training - Logistic Regression](#model-training---ogistic-regression)
  - [Model evaluation - Accuracy Score](#model-evaluation---accuracy-score)
  - [Building a predictive system](#building-a-predictive-system)

## Project Overview

## Dataset

## Machine Learning Prediction
### Install dependencies
sets up the necessary imports and initial steps for a machine learning workflow.

### Data collection and processing
- Load the breast cancer dataset, convert it into a pandas DataFrame.
- Add a target column for class labels (benign or malignant).
- Inspect the DataFrame by displaying its shape, summary statistics, and checking for missing values.
- Analyze the distribution of the target variable,
- Group the data by label to calculate mean feature values
- Rename columns for better clarity.

### Separating features and target
- Separate the DataFrame into two components:
  - **X:** Contains all the feature columns by dropping the 'label' column.
  - **Y:** Contains only the 'label' column, which is the target variable.
- Print X and Y to display their contents.
- Prints the names of all columns in X, which will be useful later for creating a web app interface.

### Split data into train and test data
- Split the dataset into training and testing sets, allocating 80% of the data for training and 20% for testing, with a fixed random seed (3) to ensure the results are reproducible.
- Print the shapes of the original dataset and the resulting training and testing sets.
- Initialize a StandardScaler to standardize the feature values and fits the scaler on the training data.

### Model training - Logistic Regression
- Create an instance of the LogisticRegression class for binary classification and train this model using the training data (X_train and y_train).
- The fit method adjusts the model parameters to best learn the relationship between the input features and target labels in the training set.

### Model evaluation - Accuracy Score
- Make predictions on both the training and test datasets using the trained logistic regression model (lr).
- Calculate the accuracy of these predictions by comparing them to the actual labels.
- The accuracy for the training data and the test data is then printed to assess the model's performance on each dataset.

### Building a predictive system
