# app.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Initial shape:", df.shape)
    return df

# Remove duplicates
def remove_duplicates(df):
    df = df.drop_duplicates()
    print("After removing duplicates:", df.shape)
    return df

# Handle missing values with median imputation
def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print("Missing values handled.")
    return df

# Convert date columns to datetime
def convert_data_types(df):
    df['Week Start'] = pd.to_datetime(df['Week Start'])
    df['Week End'] = pd.to_datetime(df['Week End'])
    df['ZIP Code'] = df['ZIP Code'].astype(str)
    print("Date and ZIP Code columns converted.")
    return df

# Remove outliers using IQR
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f"Outliers removed in {col}")
    return df

# Normalize data using Min-Max Scaling
def normalize_columns(df, columns):
    scaler = MinMaxScaler()
    df[[col + ' (Norm)' for col in columns]] = scaler.fit_transform(df[columns])
    print("Normalization done.")
    return df

# Main driver
def main():
    filepath = "COVID_data.csv"  # adjust path as needed
    df = load_data(filepath)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = remove_outliers(df, ['Case Rate - Weekly', 'Death Rate - Weekly', 'Test Rate - Weekly'])
    df = normalize_columns(df, ['Case Rate - Weekly', 'Death Rate - Weekly', 'Test Rate - Weekly'])

    # Save cleaned dataset (optional)
    df.to_csv("Cleaned_COVID_data.csv", index=False)
    print("Cleaned data saved to 'Cleaned_COVID_data.csv'")

if __name__ == "__main__":
    # Ensure the script runs only when executed directly
    if __name__ == "__main__":
        main()
