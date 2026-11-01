import pandas as pd

# Load the dataset
df = pd.read_csv("data/network_traffic_sample.csv")

# 1. Display first five records
print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

# 2. Display last five records
print("\n========== LAST FIVE RECORDS ==========")
print(df.tail())

# 3. Check dataset dimensions
print("\n========== DATASET SHAPE ==========")
print("Rows and Columns:", df.shape)

# 4. Check column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 5. Check data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 6. Dataset information
print("\n========== DATASET INFORMATION ==========")
df.info()

# 7. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 8. Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

# 9. Check attack categories
print("\n========== ATTACK DISTRIBUTION ==========")
print(df["Label"].value_counts())

# 10. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\nDATASET INSPECTION COMPLETED!")
