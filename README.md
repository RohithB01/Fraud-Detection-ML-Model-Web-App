# Fraud Detection Machine Learning Project

This project demonstrates an end-to-end machine learning pipeline to detect fraudulent financial transactions using a large real-world dataset. The notebook performs data loading, exploratory data analysis, feature engineering, model training, evaluation, and prepares a logistic regression model for deployment.

## Dataset

The dataset used in this project is the [Fraud Detection Dataset by Aman Ali Siddiqui](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) from Kaggle. It contains millions of transaction records with features related to transaction types, amounts, account balances, and fraud labels.

## Project Highlights

- Download and load dataset programmatically using `kagglehub`.
- Exploratory Data Analysis (EDA) using `pandas`, `seaborn`, and `matplotlib` with dark-themed plots.
- Feature engineering by calculating balance differences to expose suspicious behaviors.
- Preprocessing pipeline using `ColumnTransformer`:
  - Standard scaling for numeric features.
  - One-hot encoding for categorical variables.
- Logistic Regression model with balanced class weights to address class imbalance.
- Model evaluation using classification report, confusion matrix, and ROC AUC metrics.
- Model pipeline saved with `joblib` for further use.

## Requirements

- Python 3.11+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- kagglehub (for dataset download)

Install dependencies with:


## How to Use

1. Clone or download this repository.
2. Run the Jupyter notebook `Fraud_Detection_ML_Model_-_Web_App.ipynb`.
3. The notebook will download the dataset, perform analysis, train the model, and save the pipeline.
4. Use the trained model for predictions or integrate it into a deployment pipeline.

## Project Structure

- `Fraud_Detection_ML_Model_-_Web_App.ipynb`: The full Jupyter notebook containing all project steps.

## Acknowledgments

Dataset from [Aman Ali Siddiqui - Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset).

---

Feel free to open issues or contribute improvements!
