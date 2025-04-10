import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Load the data
data = pd.read_csv("C:/Users/mn440/Downloads/walmart.csv")

# Display first 5 rows
print("First 5 rows of the data:")
print(data.head())

# Shape of the dataset
print("\nShape of the dataset:", data.shape)

# Columns and data types
print("\nData Types:")
print(data.dtypes)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# Column names
print("\nColumn Names:")
print(data.columns.tolist())

# Unique values in each column
print("\nUnique Values in Each Column:")
for col in data.columns:
    print(f"{col}: {data[col].nunique()} unique values")

# Count of each unique value per column (for object type)
for col in data.select_dtypes(include='object').columns:
    print(f"\nValue counts for {col}:")
    print(data[col].value_counts())

# Info summary
print("\nData Info:")
print(data.info())

# Check for duplicates
print("\nNumber of duplicate rows:", data.duplicated().sum())

# Drop duplicates (if needed)
data = data.drop_duplicates()

# Correlation matrix
print("\nCorrelation Matrix:")
correlation = data.corr(numeric_only=True)
print(correlation)

# Heatmap of correlation
plt.figure(figsize=(10, 6))
sn.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for numerical columns
sn.pairplot(data.select_dtypes(include=[np.number]))
plt.suptitle("Pairplot of Numerical Features", y=1.02)
plt.show()

# Histograms
data.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numeric Columns")
plt.tight_layout()
plt.show()

# Boxplots to detect outliers
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sn.boxplot(x=data[col], color='lightblue')
    plt.title(f'Boxplot of {col}')
    plt.show()

# Time series analysis (if 'Date' column exists)
if 'Date' in data.columns:
    print("\nConverting 'Date' to datetime...")
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    print("Index set to Date.")

    plt.figure(figsize=(12, 6))
    data.resample('M').sum(numeric_only=True).plot()
    plt.title('Monthly Sales Trends')
    plt.ylabel('Total')
    plt.show()

# Groupby Store
if 'Store' in data.columns:
    store_sales = data.groupby('Store').sum(numeric_only=True)
    print("\nSales per Store:")
    print(store_sales)

    plt.figure(figsize=(10, 5))
    store_sales['Weekly_Sales'].plot(kind='bar', color='orange')
    plt.title('Weekly Sales per Store')
    plt.xlabel('Store')
    plt.ylabel('Sales')
    plt.show()

# Group by Department if available
if 'Dept' in data.columns:
    dept_sales = data.groupby('Dept').sum(numeric_only=True)
    print("\nSales per Department:")
    print(dept_sales)

    plt.figure(figsize=(12, 5))
    dept_sales['Weekly_Sales'].plot(kind='bar', color='green')
    plt.title('Weekly Sales per Department')
    plt.xlabel('Department')
    plt.ylabel('Sales')
    plt.show()

# Weekly Sales over time (line plot)
if 'Weekly_Sales' in data.columns:
    data['Weekly_Sales'].plot(figsize=(15, 5), title='Weekly Sales Over Time', color='purple')
    plt.ylabel('Weekly Sales')
    plt.show()

# Average sales per holiday
if 'IsHoliday' in data.columns:
    holiday_avg = data.groupby('IsHoliday')['Weekly_Sales'].mean()
    print("\nAverage Weekly Sales by Holiday Status:")
    print(holiday_avg)

    holiday_avg.plot(kind='bar', color='red')
    plt.title('Average Weekly Sales: Holiday vs Non-Holiday')
    plt.xlabel('IsHoliday')
    plt.ylabel('Avg Weekly Sales')
    plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
    plt.show()

# Trend by Store and Date
if 'Store' in data.columns and 'Weekly_Sales' in data.columns:
    plt.figure(figsize=(15, 7))
    for store in data['Store'].unique()[:5]:  # First 5 stores
        store_data = data[data['Store'] == store]
        plt.plot(store_data.index, store_data['Weekly_Sales'], label=f'Store {store}')
    plt.legend()
    plt.title('Weekly Sales for First 5 Stores Over Time')
    plt.ylabel('Weekly Sales')
    plt.xlabel('Date')
    plt.show()

# Detecting outliers numerically
print("\nOutlier Detection (IQR Method):")
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} outliers")

# Correlation with target
if 'Weekly_Sales' in data.columns:
    sales_corr = correlation['Weekly_Sales'].sort_values(ascending=False)
    print("\nCorrelation with Weekly_Sales:")
    print(sales_corr)

# Scatter plots for top correlated features
top_corr = sales_corr.index[1:4]  # excluding 'Weekly_Sales' itself
for col in top_corr:
    plt.figure(figsize=(6, 4))
    sn.scatterplot(x=data[col], y=data['Weekly_Sales'])
    plt.title(f'Weekly Sales vs {col}')
    plt.xlabel(col)
    plt.ylabel('Weekly Sales')
    plt.show()

# Aggregation
print("\nAggregated Data (mean per Store):")
print(data.groupby('Store').mean(numeric_only=True))

# Sorting stores by average weekly sales
avg_sales = data.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
print("\nTop 5 Stores by Average Weekly Sales:")
print(avg_sales.head())

# Plotting top 5 stores
avg_sales.head().plot(kind='bar', color='teal')
plt.title('Top 5 Stores by Average Weekly Sales')
plt.ylabel('Average Sales')
plt.show()

# Distribution of Weekly Sales
plt.figure(figsize=(8, 5))
sn.histplot(data['Weekly_Sales'], kde=True, color='steelblue')
plt.title('Distribution of Weekly Sales')
plt.show()

# Cumulative sales trend
data['Cumulative_Sales'] = data['Weekly_Sales'].cumsum()
plt.figure(figsize=(10, 5))
plt.plot(data['Cumulative_Sales'], color='darkblue')
plt.title('Cumulative Weekly Sales')
plt.ylabel('Sales')
plt.show()

# Saving cleaned dataset
data.reset_index(inplace=True)
data.to_csv("cleaned_walmart_data.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_walmart_data.csv'")
