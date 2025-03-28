# Decoding phone usage patterns in India

# Problem Statement
Design a system to analyze mobile device usage and user behavior by using a dataset containing user information and device statistics. The project aims to preprocess and clean the data, apply machine learning and clustering techniques, and build models to classify primary use and identify distinct usage patterns. The final application will be an interactive interface deployed with Streamlit, which will include EDA visualizations and model results.

# Step 1: Data Collection & Understanding
import pandas as pd

  Load the dataset

  file_path = "Downloads\phone_usage_india.csv"  # Update if the path is different
  df = pd.read_csv(file_path)

  Display basic dataset info

  print("\n First 5 Rows of Dataset:")
  print(df.head())

  print("\n Dataset Info:")
  print(df.info())

  print("\n Summary Statistics:")
  print(df.describe(include="all"))

  print("\n Missing Values in Each Column:")
  print(df.isnull().sum())

  print("\n Duplicate Rows Count:")
  print(df.duplicated().sum())
