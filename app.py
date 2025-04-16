import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Initial shape:", df.shape)
    return df

def remove_duplicates(df):
    df = df.drop_duplicates()
    print("After removing duplicates:", df.shape)
    return df

def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print("Missing values handled.")
    return df

def convert_data_types(df):
    df['Week Start'] = pd.to_datetime(df['Week Start'])
    df['Week End'] = pd.to_datetime(df['Week End'])
    df['ZIP Code'] = df['ZIP Code'].astype(str)
    print("Date and ZIP Code columns converted.")
    return df

# Objective 2
def plot_weekly_trends(df):
    weekly_data = df.groupby('Week Start')[[
        'Cases - Weekly', 'Deaths - Weekly', 'Tests - Weekly'
    ]].sum().reset_index()

    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))
    plt.plot(weekly_data['Week Start'], weekly_data['Cases - Weekly'], label='Cases', marker='o')
    plt.plot(weekly_data['Week Start'], weekly_data['Deaths - Weekly'], label='Deaths', marker='s')
    plt.plot(weekly_data['Week Start'], weekly_data['Tests - Weekly'], label='Tests', marker='^')

    plt.title('Weekly COVID-19 Trends')
    plt.xlabel('Week Start')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/weekly_line.png", dpi=300)
    plt.show()

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
    plt.savefig("images/weekly_area.png", dpi=300)
    plt.show()

# Objective 3
def correlation_analysis(df):
    corr_df = df[['Cases - Weekly', 'Deaths - Weekly', 'Tests - Weekly']]
    
    correlation_matrix = corr_df.corr()

    print("Correlation Matrix:\n", correlation_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap - Cases, Deaths, Tests")
    plt.tight_layout()
    plt.savefig("images/correlation_heatmap.png", dpi=300)
    plt.show()
    plt.close()
    
# Objective 4 
def zip_code_severity_analysis(df):
    zip_summary = df.groupby('ZIP Code')[[
        'Cases - Weekly', 'Deaths - Weekly', 'Tests - Weekly'
    ]].sum().reset_index()

    top_cases = zip_summary.sort_values(by='Cases - Weekly', ascending=False).head(10)
    top_deaths = zip_summary.sort_values(by='Deaths - Weekly', ascending=False).head(10)
    top_tests = zip_summary.sort_values(by='Tests - Weekly', ascending=False).head(10)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_cases, x='ZIP Code', y='Cases - Weekly', palette='Reds_r')
    plt.title("Top 10 ZIP Codes by Total Cases")
    plt.ylabel("Total Cases")
    plt.xlabel("ZIP Code")
    plt.tight_layout()
    plt.savefig("images/top_zip_cases.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_deaths, x='ZIP Code', y='Deaths - Weekly', palette='Greys')
    plt.title("Top 10 ZIP Codes by Total Deaths")
    plt.ylabel("Total Deaths")
    plt.xlabel("ZIP Code")
    plt.tight_layout()
    plt.savefig("images/top_zip_deaths.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_tests, x='ZIP Code', y='Tests - Weekly', palette='Blues')
    plt.title("Top 10 ZIP Codes by Total Tests")
    plt.ylabel("Total Tests")
    plt.xlabel("ZIP Code")
    plt.tight_layout()
    plt.savefig("images/top_zip_tests.png", dpi=300)
    plt.show()
    plt.close()

def plot_geo_heatmap(df):
    df[['Longitude', 'Latitude']] = df['ZIP Code Location'].str.extract(r'POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)').astype(float)

    zip_summary = df.groupby(['ZIP Code', 'Longitude', 'Latitude'])[
        ['Cases - Weekly', 'Deaths - Weekly', 'Tests - Weekly']
    ].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=zip_summary,
        x='Longitude', y='Latitude',
        size='Cases - Weekly',
        hue='Cases - Weekly',
        palette='Reds',
        sizes=(20, 300),
        legend='brief'
    )
    plt.title("COVID-19 Severity by ZIP Code (Geo Heatmap)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/geo_zip_heatmap_cases.png", dpi=300)
    plt.show()
    plt.close()

# Objective 5 
def plot_case_rate_outliers(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['Case Rate - Weekly'], color='skyblue')
    plt.title("Box Plot - Outlier Detection(Weekly Case Rates)")
    plt.ylabel("Case Rate - Weekly")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/case_rate_outliers_boxplot.png", dpi=300)
    plt.show()
    plt.close()

# Objective 6
def perform_eda_visualizations(df):
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Cases - Weekly'], bins=30, kde=False, color='skyblue')
    plt.title("Distribution of Weekly Cases")
    plt.xlabel("Cases - Weekly")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("images/hist_cases_weekly.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Tests - Weekly', y='Cases - Weekly', alpha=0.6)
    plt.title("Scatter Plot: Weekly Tests vs Weekly Cases")
    plt.xlabel("Tests - Weekly")
    plt.ylabel("Cases - Weekly")
    plt.tight_layout()
    plt.savefig("images/scatter_tests_vs_cases.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['Deaths - Weekly'], fill=True, color='lightcoral')
    plt.title("Density Plot of Weekly Deaths")
    plt.xlabel("Deaths - Weekly")
    plt.tight_layout()
    plt.savefig("images/kde_deaths_weekly.png", dpi=300)
    plt.show()
    plt.close()

    # Summary statistics
    summary = df[['Cases - Weekly', 'Deaths - Weekly', 'Tests - Weekly']].describe()
    print("🔍 Summary Statistics:\n", summary)

# Main driver
def main():
    filepath = "data/COVID_data.csv"  # adjust path as needed
    df = load_data(filepath)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df.to_csv("data/Cleaned_COVID_data.csv", index=False)
    
    # Objective 2 
    plot_weekly_trends(df)

    # Objective 3
    correlation_analysis(df)

    # Objective 4
    zip_code_severity_analysis(df)
    plot_geo_heatmap(df)

    # Objective 5
    plot_case_rate_outliers(df)

    # Objective 6
    perform_eda_visualizations(df)

if __name__ == "__main__":
    if __name__ == "__main__":
        main()