# Decoding phone usage patterns in India

# Problem Statement
Design a system to analyze mobile device usage and user behavior by using a dataset containing user information and device statistics. The project aims to preprocess and clean the data, apply machine learning and clustering techniques, and build models to classify primary use and identify distinct usage patterns. The final application will be an interactive interface deployed with Streamlit, which will include EDA visualizations and model results.

# Step 1: Data Collection & Understanding
    import pandas as pd
   
    # Load the dataset
    file_path = "Downloads\phone_usage_india.csv"  # Update if the path is different
    df = pd.read_csv(file_path)

    # Display basic dataset info
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

    
    
    import pandas as pd
    import numpy as np

    # Load dataset (update the path if needed)
    file_path = "Downloads/phone_usage_india.csv"
    df = pd.read_csv(file_path)

    # 1. Drop 'User ID' (not useful for ML)
    df.drop(columns=['User ID'], inplace=True)

    # 2. Handling Outliers using IQR Method
    def remove_outliers(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    numeric_columns = [
        "Age", "Screen Time (hrs/day)", "Data Usage (GB/month)",
        "Calls Duration (mins/day)", "Number of Apps Installed",
        "Social Media Time (hrs/day)", "E-commerce Spend (INR/month)",
        "Streaming Time (hrs/day)", "Gaming Time (hrs/day)", "Monthly Recharge Cost (INR)"
    ]

    df = remove_outliers(df, numeric_columns)

    # 3. Encoding Categorical Variables
    categorical_columns = ["Gender", "Location", "Phone Brand", "OS", "Primary Use"]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Save the cleaned dataset
    df.to_csv("cleaned_phone_usage_data.csv", index=False)

    print("Data cleaning complete! Cleaned dataset saved as 'cleaned_phone_usage_data.csv'.")


# Step 2: Data Cleaning & Preprocessing

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load the cleaned dataset
    df = pd.read_csv("cleaned_phone_usage_data.csv")

    # Set style
    sns.set(style="whitegrid")

    # 1. Feature Distributions
    numerical_features = ['Age', 'Screen Time (hrs/day)', 'Data Usage (GB/month)',
                          'Calls Duration (mins/day)', 'Number of Apps Installed',
                          'Social Media Time (hrs/day)', 'Streaming Time (hrs/day)',
                          'Gaming Time (hrs/day)', 'Monthly Recharge Cost (INR)']

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

    # 2. Categorical Features Count Plots
    categorical_features = [
        "Gender_Male", "Gender_Other",  # Gender categories
        "Location_Bangalore", "Location_Chennai", "Location_Delhi", 
        "Location_Hyderabad", "Location_Jaipur", "Location_Kolkata", 
        "Location_Lucknow", "Location_Mumbai", "Location_Pune",  # Locations
        "Phone Brand_Google Pixel", "Phone Brand_Motorola", "Phone Brand_Nokia", 
        "Phone Brand_OnePlus", "Phone Brand_Oppo", "Phone Brand_Realme", 
        "Phone Brand_Samsung", "Phone Brand_Vivo", "Phone Brand_Xiaomi",  # Brands
        "Primary Use_Entertainment", "Primary Use_Gaming", 
        "Primary Use_Social Media", "Primary Use_Work"  # Primary Use
    ]

    plt.figure(figsize=(12, 8))

    for i, col in enumerate(categorical_features[:4], 1):  # Plot only the first 4
        plt.subplot(2, 2, i)
        sns.countplot(data=df, x=col, palette="Set2")
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[numerical_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # 4. Box Plots for Outliers
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

    
