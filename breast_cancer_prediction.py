# ============================================================
# BREAST CANCER PREDICTION PROJECT 
# Student: Nakaddu Charity
# Reg No: S24B38/027, B30496
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import joblib
from graphviz import Digraph
from IPython.display import Image

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================
output_dir = "Breast_Cancer_Project_Outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ============================================================
# 3. LOAD DATASET (SCIKIT-LEARN)
# ============================================================
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("Dataset Loaded Successfully\n")
display(df.head())
print(df.info())
print(df["target"].value_counts())

# ============================================================
# 4. SAVE DATASET FOR SUBMISSION
# ============================================================
dataset_path = f"{output_dir}/breast_cancer_dataset.csv"
df.to_csv(dataset_path, index=False)
print(f"Dataset saved for submission: {dataset_path}")

# ============================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# 5.1 Missing Values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# 5.2 Summary Statistics
summary_stats = df.describe()
summary_stats.to_csv(f"{output_dir}/summary_statistics.csv")

# 5.3 Target Distribution Plot
sns.countplot(x="target", data=df)
plt.title("Target Distribution (0=Malignant,1=Benign)")
plt.savefig(f"{output_dir}/target_distribution.png")
plt.show()

# 5.4 Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.show()

# 5.5 Outlier Detection (IQR)
outlier_report = {}
for col in df.select_dtypes(include=np.number).columns[:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    outlier_report[col] = len(outliers)
outlier_df = pd.DataFrame.from_dict(outlier_report, orient='index', columns=['Outlier Count'])
outlier_df.to_csv(f"{output_dir}/outlier_report.csv")

# ============================================================
# 6. FEATURE SELECTION (MUTUAL INFORMATION)
# ============================================================
X = df.drop("target", axis=1)
y = df["target"]

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
mi_scores.to_csv(f"{output_dir}/mutual_information_scores.csv")

top_features = mi_scores.index[:15]
X_selected = X[top_features]
pd.Series(top_features).to_csv(f"{output_dir}/top_15_features.csv")

print("Top 15 Features Selected:")
print(top_features)

# ============================================================
# 7. SPLIT DATA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 8. STANDARDIZE FEATURES
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, f"{output_dir}/scaler.pkl")

# ============================================================
# 9. TRAIN MODELS & HYPERPARAMETER TUNING
# ============================================================
# Random Forest
rf = RandomForestClassifier(random_state=42)
param_grid = {"n_estimators":[100,200], "max_depth":[None,5,10], "min_samples_split":[2,5]}
grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy")
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
joblib.dump(best_rf, f"{output_dir}/rf_model.pkl")
print("Best RF Parameters:", grid_rf.best_params_)

# Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_scaled, y_train)
joblib.dump(lr, f"{output_dir}/lr_model.pkl")

# ============================================================
# 10. MODEL EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"{name} ROC AUC: {roc:.4f}")

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).to_csv(f"{output_dir}/{name}_classification_report.csv")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"{output_dir}/{name}_confusion_matrix.png")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} AUC={roc:.2f}")
    plt.plot([0,1],[0,1],"k--")
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.savefig(f"{output_dir}/{name}_roc_curve.png")
    plt.show()

    return acc, roc

# Evaluate
acc_rf, roc_rf = evaluate_model(best_rf, X_test_scaled, y_test, "RandomForest")
acc_lr, roc_lr = evaluate_model(lr, X_test_scaled, y_test, "LogisticRegression")

# ============================================================
# 10.1 K-FOLD CROSS VALIDATION
# ============================================================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf, X_selected, y, cv=kf, scoring="accuracy")
print("\nRandom Forest 5-Fold CV Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# ============================================================
# 11. CREATE WORKFLOW FLOWCHART
# ============================================================
flow = Digraph(comment='Breast Cancer Prediction Workflow', format='png')
flow.node('A', 'Load Dataset')
flow.node('B', 'EDA')
flow.node('C', 'Feature Selection (MI)')
flow.node('D', 'Split & Scale Data')
flow.node('E', 'Train Models (RF, LR)')
flow.node('F', 'Evaluate Models')
flow.node('G', 'K-Fold CV')
flow.node('H', 'Save Outputs (CSV, Plots, Models)')
flow.node('I', 'Create ZIP Submission')
flow.edges(['AB','BC','CD','DE','EF','FG','FH','HI'])
flow_path = f"{output_dir}/project_workflow_flowchart"
flow.render(flow_path, view=False)
print(f"Workflow flowchart saved as {flow_path}.png")
Image(filename=f"{flow_path}.png")

# ============================================================
# 12. CREATE COMPLETE SUBMISSION ZIP
# ============================================================
zip_filename = "Breast_Cancer_Project_Submission_Nakaddu_Charity"
shutil.make_archive(zip_filename, "zip", root_dir=output_dir)
print(f"\nSubmission ZIP created successfully: {zip_filename}.zip")

# Colab download
try:
    from google.colab import files
    files.download(f"{zip_filename}.zip")
    print("ZIP ready for download.")
except ImportError:
    print("Run this in Google Colab to enable direct download.")
