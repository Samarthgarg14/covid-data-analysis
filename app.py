import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

# Objective 2
def plot_weekly_trends(df):
    # Grouping by week number (or use 'Week Start' if you want exact dates)
    weekly_data = df.groupby('Week Number')[[
        'Cases - Weekly', 'Deaths - Weekly', 'Tests - Weekly'
    ]].sum().reset_index()

    # Set style
    sns.set(style="whitegrid")

    # Line Plot
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_data['Week Number'], weekly_data['Cases - Weekly'], label='Cases', marker='o')
    plt.plot(weekly_data['Week Number'], weekly_data['Deaths - Weekly'], label='Deaths', marker='s')
    plt.plot(weekly_data['Week Number'], weekly_data['Tests - Weekly'], label='Tests', marker='^')

    plt.title('Weekly COVID-19 Trends')
    plt.xlabel('Week Number')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig("weekly_line.png", dpi=300)
    plt.show()

    # Stacked Area Plot
    plt.figure(figsize=(12, 6))
    plt.stackplot(weekly_data['Week Number'],
                  weekly_data['Cases - Weekly'],
                  weekly_data['Deaths - Weekly'],
                  weekly_data['Tests - Weekly'],
                  labels=['Cases', 'Deaths', 'Tests'],
                  alpha=0.7)

    plt.title('Weekly COVID-19 Stacked Area Chart')
    plt.xlabel('Week Number')
    plt.ylabel('Total Count')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("weekly_area.png", dpi=300)
    plt.show()

# Objective 3
def correlation_analysis(df):
    # Select relevant columns
    corr_df = df[['Cases - Weekly', 'Deaths - Weekly', 'Tests - Weekly']]
    
    # Compute correlation matrix
    correlation_matrix = corr_df.corr()

    # Print matrix
    print("Correlation Matrix:\n", correlation_matrix)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap - Cases, Deaths, Tests")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300)
    plt.show()
    plt.close()
    print(df[['Cases - Weekly', 'Deaths - Weekly']].describe())
    print(df[['Cases - Weekly', 'Deaths - Weekly']].head(10))


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

    # Objective 2 
    plot_weekly_trends(df)

    # Objective 3
    correlation_analysis(df)

if __name__ == "__main__":
    if __name__ == "__main__":
        main()