# ============================
# IMPORTS
# ============================
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ============================
# CREATE OUTPUT DIRECTORIES
# ============================
os.makedirs("results", exist_ok=True)
os.makedirs("ml_model", exist_ok=True)

# ============================
# LOAD DATA
# ============================
df = pd.read_csv("student_data.csv")

# ============================
# DATA CLEANING
# ============================
df = df.dropna(subset=["performance"])
df = df[df["performance"].astype(str).str.strip() != ""]
df["performance"] = df["performance"].str.strip().str.title()

print("Class distribution:\n", df["performance"].value_counts())

# ============================
# ADD CONTROLLED NOISE (REALISM)
# ============================
np.random.seed(42)

score_cols = [
    "assignments_score",
    "midterm_score",
    "final_score"
]

for col in score_cols:
    noise = np.random.normal(0, 3, len(df))
    df[col] = (df[col] + noise).clip(0, 100).round()

noise_fraction = 0.05
n_noisy = int(len(df) * noise_fraction)
indices = np.random.choice(df.index, n_noisy, replace=False)
labels = df["performance"].unique()

for idx in indices:
    current = df.at[idx, "performance"]
    df.at[idx, "performance"] = np.random.choice(
        [l for l in labels if l != current]
    )

# ============================
# FEATURES & TARGET
# ============================
X = df[
    ["attendance", "assignments_score", "midterm_score",
     "final_score", "study_hours"]
]

le = LabelEncoder()
y = le.fit_transform(df["performance"])

# ============================
# TRAIN / TEST SPLIT (STRATIFIED)
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ============================
# MODEL TRAINING (RANDOM FOREST)
# ============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# ============================
# EVALUATION
# ============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    zero_division=0
))

# ----------------------------
# Classification report DF
# ----------------------------
report_dict = classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    output_dict=True
)
df_report = pd.DataFrame(report_dict).transpose()
class_df = df_report.loc[le.classes_, ["precision", "recall", "f1-score"]]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

