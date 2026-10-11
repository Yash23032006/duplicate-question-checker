
import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocessing import clean_text


# 1. Load dataset
DATA_PATH = "data/train.csv"
df = pd.read_csv(DATA_PATH)

df = df[
    ["question1", "question2", "is_duplicate"]
].dropna()

# Reproducible sample for a laptop
df = df.sample(
    n=min(50000, len(df)),
    random_state=42
).reset_index(drop=True)

df["question1"] = df["question1"].apply(clean_text)
df["question2"] = df["question2"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df[["question1", "question2"]],
    df["is_duplicate"].astype(int),
    test_size=0.2,
    random_state=42,
    stratify=df["is_duplicate"]
)

# 2. TF-IDF feature extraction
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

train_texts = pd.concat([
    X_train["question1"],
    X_train["question2"]
])

vectorizer.fit(train_texts)

def make_features(q1, q2):
    v1 = vectorizer.transform(q1)
    v2 = vectorizer.transform(q2)

    # Pairwise features:
    # absolute difference and shared weighted terms
    difference = abs(v1 - v2)
    product = v1.multiply(v2)

    return hstack(
        [difference, product],
        format="csr"
    )

Xtr = make_features(
    X_train["question1"], X_train["question2"]
)
Xte = make_features(
    X_test["question1"], X_test["question2"]
)

# 3. Train and compare models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Linear SVM": LinearSVC()
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(Xtr, y_train)

    predictions = model.predict(Xte)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    if name == "Logistic Regression":
        scores = model.predict_proba(Xte)[:, 1]
    else:
        scores = model.decision_function(Xte)

    metrics["roc_auc"] = roc_auc_score(y_test, scores)

    results[name] = metrics
    trained_models[name] = model

    print(name, metrics)

# 4. Save both models and the shared vectorizer
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "vectorizer": vectorizer,
        "models": trained_models,
        "metrics": results
    },
    "models/duplicate_model.joblib"
)

print("\nModels saved successfully!")
