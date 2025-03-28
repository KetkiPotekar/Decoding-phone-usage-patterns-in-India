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

 # Step 2: Data Cleaning & Preprocessing   
    
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


# Step 3: Exploratory Data Analysis (EDA)

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

<img width="547" alt="Image" src="https://github.com/user-attachments/assets/ed982a6d-c013-40eb-b638-a511ea709fbe" />
    
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

<img width="313" alt="Image" src="https://github.com/user-attachments/assets/442f4b73-beff-46b4-bb06-7cdd3c90e371" />

    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[numerical_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

<img width="554" alt="Image" src="https://github.com/user-attachments/assets/7e5946e6-853b-4e92-bac7-1dbc9423682b" />

    # 4. Box Plots for Outliers
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

<img width="554" alt="Image" src="https://github.com/user-attachments/assets/a8d095a0-1fe8-49ef-b9ee-4247143caf71" />

    #Distribution of Numerical Variables
    df.hist(figsize=(12, 10), bins=30, edgecolor='black')
    plt.suptitle("Distribution of Numerical Features", fontsize=16)
    plt.show()

<img width="483" alt="Image" src="https://github.com/user-attachments/assets/cf4ca816-63f3-41d4-8bd8-5c8ce1960ae8" />

    # Outlier Detection (Boxplots)
    plt.figure(figsize=(15, 8))
    df.boxplot(rot=90)  # Rotate labels for readability
    plt.title("Boxplot for Outlier Detection")
    plt.show()

<img width="604" alt="Image" src="https://github.com/user-attachments/assets/262873bd-adbb-4b30-ac57-5e34edcd91c8" />

    # Check Correlations Between Numerical Features
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.show()

<img width="696" alt="Image" src="https://github.com/user-attachments/assets/59c06187-072d-48f6-970d-d42fd0c837fe" />

     
# Step 4: Feature Engineering & Preprocessing

    # Handle Outliers

    from scipy.stats.mstats import winsorize
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
       df[col] = winsorize(df[col], limits=[0.01, 0.01])  # Trim 1% extreme values

    # Normalize/Standardize Numerical Features

    from sklearn.preprocessing import StandardScaler
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])  

# Step 5: Train Machine Learning Models    

    # Splitting Data for Training & Testing

    # Define target variable (multi-class)
    target_columns = ['Primary Use_Entertainment', 'Primary Use_Gaming', 'Primary Use_Social Media', 'Primary Use_Work']

    # Define features (exclude target columns)
    X = df.drop(columns=target_columns)
    y = df[target_columns]  # Multi-label classification

    # Train-test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    
    # training one model

    # Find columns related to Primary Use
    primary_use_columns = ['Primary Use_Entertainment', 'Primary Use_Gaming', 'Primary Use_Social Media', 'Primary Use_Work']

    # Convert One-Hot Encoding back to a single categorical column
    y = df[primary_use_columns].idxmax(axis=1)  # Get the column with max value
    y = y.str.replace("Primary Use_", "")  # Remove prefix

    # Split again
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # 1: Train a Logistic Regression Model

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Initialize Logistic Regression

    logreg = LogisticRegression(max_iter=1000, multi_class='ovr', solver='liblinear')
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")

    # Logistic Regression Accuracy: 0.3988


    # Training and comparing 5 models 

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='ovr', solver='liblinear'),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True)
    }

    # Dictionary to store results
    results = {}

    # Train and evaluate models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Store results
        results[name] = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm
        }
    
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(report)
        print("-" * 50)

    # Plot Confusion Matrices
    plt.figure(figsize=(15, 10))
    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(2, 3, i)
        sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()

<img width="656" alt="Image" src="https://github.com/user-attachments/assets/3330d57a-c5a3-47fd-a52f-54631d9f8f44" />

    # ROC Curve Plot
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Binarize the target variable for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])  # Adjust class labels if needed

    plt.figure(figsize=(10, 7))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)
    
        for i in range(y_test_bin.shape[1]):  # Loop through each class
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (Class {i}) - AUC: {roc_auc:.2f}')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal reference line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-class Classification')
    plt.legend()
    plt.show()

<img width="548" alt="Image" src="https://github.com/user-attachments/assets/6e9dd895-d6c7-472a-b8ac-a58d95fda1e1" />

# Model Comparison and Best Choice
Results indicate poor performance across all models, with accuracy below 40% and imbalanced predictions (most models predict "Entertainment" for almost everything). Here's a breakdown:

<img width="591" alt="Image" src="https://github.com/user-attachments/assets/ddd78f1b-96db-4a3c-901b-d32a172760a9" />

Best Model?
Right now, Gradient Boosting is the best choice because: 
✅ It achieves the best recall for "Entertainment" (98%), meaning it correctly identifies those users.
✅ It has the highest macro-average recall, meaning it does slightly better at capturing other classes.
✅ Tree-based models generally handle structured data better than linear models like Logistic Regression or SVM.

# Step 6 : Model Deployment with Streamlit

    # Designing Streamlit app

    import streamlit as st
    import numpy as np

    st.title("Phone Usage Prediction App")

    age = st.number_input("Enter your age")
    screen_time = st.number_input("Screen Time (hrs/day)")
    data_usage = st.number_input("Data Usage (GB/month)")

    if st.button("Predict"):
        input_data = np.array([[age, screen_time, data_usage]])  # Add all features
        prediction = best_model.predict(input_data)
        st.write(f"Predicted Primary Use: {prediction[0]}")

        
