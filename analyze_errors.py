
import os
import joblib
import pandas as pd

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from preprocessing import clean_text

DATA_PATH = "data/train.csv"
MODEL_PATH = "models/duplicate_model.joblib"
OUTPUT_PATH = "evaluation_results/misclassified_linear_svm.csv"

print("Loading dataset and trained model...")

df = pd.read_csv(DATA_PATH)
df = df[["question1", "question2", "is_duplicate"]].dropna()

df = df.sample(
    n=min(50000, len(df)),
    random_state=42
).reset_index(drop=True)

df["question1"] = df["question1"].apply(clean_text)
df["question2"] = df["question2"].apply(clean_text)
df["is_duplicate"] = df["is_duplicate"].astype(int)

X = df[["question1", "question2"]]
y = df["is_duplicate"]

# Reproduce the training test split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

bundle = joblib.load(MODEL_PATH)
vectorizer = bundle["vectorizer"]
model = bundle["models"]["Linear SVM"]

def make_features(q1, q2):
    v1 = vectorizer.transform(q1)
    v2 = vectorizer.transform(q2)

    return hstack(
        [abs(v1 - v2), v1.multiply(v2)],
        format="csr"
    )

features = make_features(
    X_test["question1"],
    X_test["question2"]
)

predictions = model.predict(features)
scores = model.decision_function(features)

report = pd.DataFrame({
    "question1": X_test["question1"].values,
    "question2": X_test["question2"].values,
    "actual": y_test.values,
    "predicted": predictions,
    "decision_score": scores
})

# Keep only incorrectly classified pairs.
errors = report[report["actual"] != report["predicted"]].copy()

errors["error_type"] = errors.apply(
    lambda row: (
        "False positive"
        if row["actual"] == 0
        else "False negative"
    ),
    axis=1
)

os.makedirs("evaluation_results", exist_ok=True)
errors.to_csv(OUTPUT_PATH, index=False)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nMisclassified examples:", len(errors))
print("\nError types:")
print(errors["error_type"].value_counts())

print("\nSaved error report to:", OUTPUT_PATH)
print("\nFirst 10 errors:")
print(errors.head(10).to_string(index=False))
