import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Create required folders
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("data/network_traffic_sample.csv")

# 2. Clean data
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.drop_duplicates(inplace=True)
df.dropna(subset=["Label"], inplace=True)

# 3. Prepare features and target
X = df.drop(columns=["Label"])
X = X.select_dtypes(include=[np.number])

y = (df["Label"] != "BENIGN").astype(int)

# 4. Split data into training, validation and testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# 5. Define machine learning models
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

pipelines = {}

# 6. Train and validate models
for name, classifier in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    score = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    pipelines[name] = pipeline

    print("\nMODEL:", name)
    print("Validation Accuracy:",
          accuracy_score(y_val, predictions))
    print("Validation Precision:",
          precision_score(y_val, predictions, zero_division=0))
    print("Validation Recall:",
          recall_score(y_val, predictions, zero_division=0))
    print("Validation F1 Score:", score)

# 7. Select best model using validation F1 score
best_name = max(
    pipelines,
    key=lambda name: f1_score(
        y_val,
        pipelines[name].predict(X_val),
        zero_division=0
    )
)

best_model = pipelines[best_name]

print("\nSELECTED MODEL:", best_name)

# 8. Refit selected model on training + validation data
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

best_model.fit(X_train_full, y_train_full)

# 9. Final evaluation on untouched test data
test_predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(
    y_test, test_predictions, zero_division=0
)
recall = recall_score(
    y_test, test_predictions, zero_division=0
)
f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

print("\n========== FINAL TEST RESULTS ==========")
print("Selected Model:", best_name)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# 10. Classification report
report = classification_report(
    y_test,
    test_predictions,
    target_names=["Normal", "Attack"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

# Save classification report
with open("reports/classification_report.txt", "w") as file:
    file.write("NETWORK INTRUSION DETECTION REPORT\n")
    file.write("=" * 40 + "\n")
    file.write(f"Selected Model: {best_name}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n\n")
    file.write(report)

# 11. Generate confusion matrix
cm = confusion_matrix(y_test, test_predictions)

print("\nCONFUSION MATRIX")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)

plt.title("Network Intrusion Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig("reports/confusion_matrix.png", dpi=300)
plt.close()

print("\nConfusion matrix graph saved successfully!")

# 12. Save model
joblib.dump(
    {
        "model": best_model,
        "features": X.columns.tolist(),
        "model_name": best_name
    },
    "models/intrusion_model.pkl"
)

# 13. Save test predictions as CSV
prediction_report = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": test_predictions
})

prediction_report["Actual"] = prediction_report["Actual"].map(
    {0: "Normal", 1: "Attack"}
)

prediction_report["Predicted"] = prediction_report["Predicted"].map(
    {0: "Normal", 1: "Attack"}
)

prediction_report.to_csv(
    "reports/intrusion_detection_report.csv",
    index=False
)

print("\nMODEL SAVED SUCCESSFULLY!")
print("File: models/intrusion_model.pkl")
print("Report: reports/classification_report.txt")
print("Report: reports/intrusion_detection_report.csv")
print("Graph: reports/confusion_matrix.png")
print("\nMODEL TRAINING AND EVALUATION COMPLETED!")
