# Customer Churn Prediction - Midterm Project ML 2025

## Description

Customer churn refers to the situation when a customer discontinues their subscription or relationship with a company. Predicting churn is critical for businesses because it allows them to identify customers at risk of leaving and implement strategies to improve retention, customer satisfaction, and revenue.

In this project, we use a tabular dataset containing customer information such as age, gender, tenure, usage frequency, support calls, payment delay, subscription type, contract length, total spend, and last interaction. The goal is to build a **predictive model** to classify whether a customer is likely to churn (`Churn = 1`) or not (`Churn = 0`).  

The dataset includes:

- **Training set**: Includes all features and the `churn` label for model training.
- **Testing set**: Includes all features but **does not contain the `churn` label**, used for evaluating model predictions in a real-world scenario.

This project applies **logistic regression**, **feature engineering**, and **MLflow tracking** to build and evaluate the model, demonstrating an end-to-end machine learning workflow.

---

## Project Structure

├── data/
│ ├── customer_churn_dataset-training-master.csv
│ └── customer_churn_dataset-testing-master.csv
├── notebook/
│ └── churn_analysis.ipynb # EDA, preprocessing, model training
├── src/
│ ├── train.py # Training scripts
│ └── predict.py # Prediction scripts
├── README.md



---







