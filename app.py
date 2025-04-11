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
def normalize_columns(df, cols):
    df_normalized = df.copy()

    for col in cols:
        unique_vals = df_normalized[col].nunique()

        if unique_vals > 1:
            # Apply Min-Max Scaling
            scaler = MinMaxScaler()
            df_normalized[[col + ' (Norm)']] = scaler.fit_transform(df_normalized[[col]])
        else:
            # Copy the column as-is if only one unique value
            df_normalized[col + ' (Norm)'] = df_normalized[col]
            print(f"⚠️ Skipped normalization on '{col}' due to constant value.")

    return df_normalized



# Objective 2
def plot_weekly_trends(df):
    # Grouping by week number (or use 'Week Start' if you want exact dates)
    weekly_data = df.groupby('Week Start')[[
        'Cases - Weekly', 'Deaths - Weekly', 'Tests - Weekly'
    ]].sum().reset_index()

    # Set style
    sns.set(style="whitegrid")

    # Line Plot
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_data['Week Start'], weekly_data['Cases - Weekly'], label='Cases', marker='o')
    plt.plot(weekly_data['Week Start'], weekly_data['Deaths - Weekly'], label='Deaths', marker='s')
    plt.plot(weekly_data['Week Start'], weekly_data['Tests - Weekly'], label='Tests', marker='^')

    plt.title('Weekly COVID-19 Trends')
    plt.xlabel('Week Start')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig("weekly_line.png", dpi=300)
    plt.show()

    # Stacked Area Plot
    plt.figure(figsize=(12, 6))
    plt.stackplot(weekly_data['Week Start'],
                  weekly_data['Cases - Weekly'],
                  weekly_data['Deaths - Weekly'],
                  weekly_data['Tests - Weekly'],
                  labels=['Cases', 'Deaths', 'Tests'],
                  alpha=0.7)

    plt.title('Weekly COVID-19 Stacked Area Chart')
    plt.xlabel('Week Start')
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
    
# Objective 4 
def zip_code_severity_analysis(df):
    # Group by ZIP and sum the metrics
    zip_summary = df.groupby('ZIP Code')[[
        'Cases - Weekly', 'Deaths - Weekly', 'Tests - Weekly'
    ]].sum().reset_index()

    # Sort and get top 10 for each
    top_cases = zip_summary.sort_values(by='Cases - Weekly', ascending=False).head(10)
    top_deaths = zip_summary.sort_values(by='Deaths - Weekly', ascending=False).head(10)
    top_tests = zip_summary.sort_values(by='Tests - Weekly', ascending=False).head(10)

    # Set style
    sns.set(style="whitegrid")

    # Plot: Top 10 ZIPs by Cases
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_cases, x='ZIP Code', y='Cases - Weekly', palette='Reds_r')
    plt.title("Top 10 ZIP Codes by Total Cases")
    plt.ylabel("Total Cases")
    plt.xlabel("ZIP Code")
    plt.tight_layout()
    plt.savefig("top_zip_cases.png", dpi=300)
    plt.close()

    # Plot: Top 10 ZIPs by Deaths
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_deaths, x='ZIP Code', y='Deaths - Weekly', palette='Greys')
    plt.title("Top 10 ZIP Codes by Total Deaths")
    plt.ylabel("Total Deaths")
    plt.xlabel("ZIP Code")
    plt.tight_layout()
    plt.savefig("top_zip_deaths.png", dpi=300)
    plt.close()

    # Plot: Top 10 ZIPs by Tests
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_tests, x='ZIP Code', y='Tests - Weekly', palette='Blues')
    plt.title("Top 10 ZIP Codes by Total Tests")
    plt.ylabel("Total Tests")
    plt.xlabel("ZIP Code")
    plt.tight_layout()
    plt.savefig("top_zip_tests.png", dpi=300)
    plt.close()

    # print("✅ Bar plots saved for ZIP-wise total Cases, Deaths, and Tests.")

# Main driver
def main():
    filepath = "COVID_data.csv"  # adjust path as needed
    df = load_data(filepath)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    # df = remove_outliers(df, ['Case Rate - Weekly', 'Death Rate - Weekly', 'Test Rate - Weekly'])
    # df = normalize_columns(df, ['Case Rate - Weekly', 'Death Rate - Weekly', 'Test Rate - Weekly'])

    # Save cleaned dataset (optional)
    df.to_csv("Cleaned_COVID_data.csv", index=False)
    print("Cleaned data saved to 'Cleaned_COVID_data.csv'")
    # df = load_data('COVID_data.csv')
    # Objective 2 
    plot_weekly_trends(df)

    # Objective 3
    correlation_analysis(df)

    # Objective 4
    zip_code_severity_analysis(df)


if __name__ == "__main__":
    if __name__ == "__main__":
        main()