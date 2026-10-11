
import os
import joblib
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
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from preprocessing import clean_text


# 1. Load and prepare the dataset
DATA_PATH = "data/train.csv"
OUTPUT_PATH = "models/duplicate_model_char.joblib"

df = pd.read_csv(DATA_PATH)
df = df[["question1", "question2", "is_duplicate"]].dropna()

df = df.sample(
    n=min(50000, len(df)),
    random_state=42
).reset_index(drop=True)

df["question1"] = df["question1"].apply(clean_text)
df["question2"] = df["question2"].apply(clean_text)
df["is_duplicate"] = df["is_duplicate"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df[["question1", "question2"]],
    df["is_duplicate"],
    test_size=0.2,
    random_state=42,
    stratify=df["is_duplicate"]
)


# 2. Create word-level and character-level vectorizers
word_vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=30000,
    min_df=2,
    sublinear_tf=True
)

train_texts = pd.concat([
    X_train["question1"],
    X_train["question2"]
])

word_vectorizer.fit(train_texts)
char_vectorizer.fit(train_texts)


# 3. Build pairwise features
def make_features(q1, q2):
    w1 = word_vectorizer.transform(q1)
    w2 = word_vectorizer.transform(q2)

    c1 = char_vectorizer.transform(q1)
    c2 = char_vectorizer.transform(q2)

    word_features = hstack([
        abs(w1 - w2),
        w1.multiply(w2)
    ], format="csr")

    char_features = hstack([
        abs(c1 - c2),
        c1.multiply(c2)
    ], format="csr")

    return hstack([
        word_features,
        char_features
    ], format="csr")


Xtr = make_features(
    X_train["question1"],
    X_train["question2"]
)

Xte = make_features(
    X_test["question1"],
    X_test["question2"]
)


# 4. Train and compare classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} with word + character features...")

    model.fit(Xtr, y_train)
    predictions = model.predict(Xte)

    if name == "Logistic Regression":
        scores = model.predict_proba(Xte)[:, 1]
    else:
        scores = model.decision_function(Xte)

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
        ),
        "roc_auc": roc_auc_score(y_test, scores)
    }

    results[name] = metrics
    trained_models[name] = model

    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric:>10}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Non-duplicate", "Duplicate"],
        zero_division=0
    ))


# 5. Save the experiment separately
os.makedirs("models", exist_ok=True)

joblib.dump({
    "word_vectorizer": word_vectorizer,
    "char_vectorizer": char_vectorizer,
    "models": trained_models,
    "metrics": results,
    "feature_type": "word_and_character_tfidf"
}, OUTPUT_PATH)

print(f"\nExperiment saved to: {OUTPUT_PATH}")
print("Original model bundle was not overwritten.")
