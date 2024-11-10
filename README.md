MLP Classifier on Iris Dataset

This repository contains an implementation of a Multi-Layer Perceptron (MLP) Classifier applied to the popular Iris dataset for classification. This example demonstrates the use of neural networks for classifying species in a dataset based on distinct features.

Project Overview
The Iris dataset is widely used as a beginner’s dataset in machine learning, known for its simplicity and effectiveness in illustrating classification techniques. The dataset includes 150 samples of iris flowers from three species (setosa, versicolor, and virginica) with four feature measurements per sample.

This project uses Scikit-Learn’s MLPClassifier to train a neural network to classify the species based on the four provided features.

Key Steps in the Project
Data Loading: The Iris dataset is loaded using Scikit-Learn’s load_iris function.
Data Preprocessing: Features are scaled using Min-Max scaling to normalize the range.
Model Training: An MLPClassifier model is trained on the dataset.
Evaluation: Model performance is assessed using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Median Absolute Error (MedAE), Coefficient of Determination (R²), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
Code Structure
The main components of the code are as follows:

Importing Libraries: Necessary libraries such as numpy, pandas, Scikit-Learn, and others.
Data Loading and Preprocessing: Data is loaded, split into training and testing sets, and scaled.
Model Training: The neural network model is trained with one hidden layer.
Evaluation: Error metrics are calculated to evaluate model performance.

Evaluation Metrics
MAE (Mean Absolute Error): Measures average absolute errors between predicted and actual values.
MSE (Mean Squared Error): Measures average squared errors.
MedAE (Median Absolute Error): Provides the median of absolute errors.
R² (Coefficient of Determination): Indicates the proportion of the variance in the dependent variable predictable from the features.
RMSE (Root Mean Squared Error): Square root of MSE, showing average error magnitude.
MAPE (Mean Absolute Percentage Error): Average of absolute percentage errors.

Acknowledgements
This project is based on the Iris dataset, which is publicly available through the UCI Machine Learning Repository.

