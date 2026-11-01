import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create reports folder
Path("reports").mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv("data/network_traffic_sample.csv")

df.columns = df.columns.str.strip()

# Basic dataset information
total_records = len(df)
total_columns = len(df.columns)
missing_values = df.isnull().sum().sum()
duplicate_records = df.duplicated().sum()

# Label analysis
label_counts = df["Label"].value_counts()
benign_count = (df["Label"] == "BENIGN").sum()
attack_count = total_records - benign_count

# Generate written EDA report
with open("reports/EDA_Report.txt", "w") as report:
    report.write("NETWORK INTRUSION DETECTOR - EDA REPORT\n")
    report.write("=" * 45 + "\n\n")

    report.write(f"Total Records: {total_records}\n")
    report.write(f"Total Columns: {total_columns}\n")
    report.write(f"Missing Values: {missing_values}\n")
    report.write(f"Duplicate Records: {duplicate_records}\n\n")

    report.write(f"Benign Traffic: {benign_count}\n")
    report.write(f"Attack Traffic: {attack_count}\n\n")

    report.write("ATTACK TYPE DISTRIBUTION\n")
    report.write("-" * 30 + "\n")
    report.write(label_counts.to_string())
    report.write("\n\n")

    report.write("NUMERICAL DATA SUMMARY\n")
    report.write("-" * 30 + "\n")
    report.write(df.describe().to_string())

# Graph 1: Traffic distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Label", order=label_counts.index)
plt.title("Network Traffic Distribution")
plt.xlabel("Traffic Type")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("reports/traffic_distribution.png")
plt.close()

# Graph 2: Flow duration distribution
if "Flow Duration" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Flow Duration"].dropna(), bins=30, kde=True)
    plt.title("Flow Duration Distribution")
    plt.xlabel("Flow Duration")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("reports/flow_duration.png")
    plt.close()

# Graph 3: Attack vs Benign traffic
traffic_types = pd.Series({
    "BENIGN": benign_count,
    "ATTACK": attack_count
})

plt.figure(figsize=(7, 5))
traffic_types.plot(kind="bar")
plt.title("Benign Traffic vs Attack Traffic")
plt.xlabel("Traffic Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/benign_vs_attack.png")
plt.close()

# Graph 4: Correlation heatmap
numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
    plt.title("Network Traffic Feature Correlation")
    plt.tight_layout()
    plt.savefig("reports/correlation_heatmap.png")
    plt.close()

print("EXPLORATORY DATA ANALYSIS COMPLETED")
print("Total Records:", total_records)
print("Total Columns:", total_columns)
print("Benign Traffic:", benign_count)
print("Attack Traffic:", attack_count)
print("\nEDA report and graphs saved in the reports folder.")

