import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/network_traffic_sample.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Remove records without target labels
df.dropna(subset=["Label"], inplace=True)

# Separate features and target
X = df.drop(columns=["Label"])
y = (df["Label"] != "BENIGN").astype(int)

# Keep numerical features
X = X.select_dtypes(include=[np.number])

# Divide data into training and temporary sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Divide temporary data into validation and testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("PREPROCESSING COMPLETED")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nFeature columns:")
print(X.columns.tolist())
