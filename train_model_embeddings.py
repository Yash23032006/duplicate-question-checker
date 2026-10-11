
import os
import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
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


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
DATA_PATH = "data/train.csv"
OUTPUT_PATH = "models/duplicate_model_embeddings.joblib"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_SIZE = 50000
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load and prepare the dataset
# --------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)
df = df[["question1", "question2", "is_duplicate"]].dropna()

df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=RANDOM_STATE
).reset_index(drop=True)

df["question1"] = df["question1"].apply(clean_text)
df["question2"] = df["question2"].apply(clean_text)
df["is_duplicate"] = df["is_duplicate"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df[["question1", "question2"]],
    df["is_duplicate"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["is_duplicate"]
)

print(f"Training pairs: {len(X_train):,}")
print(f"Testing pairs:  {len(X_test):,}")


# --------------------------------------------------
# 3. Load the pretrained sentence embedding model
# --------------------------------------------------
print("\nLoading sentence embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# --------------------------------------------------
# 4. Encode each unique question only once
# --------------------------------------------------
print("\nGenerating sentence embeddings...")

all_questions = pd.concat(
    [df["question1"], df["question2"]],
    ignore_index=True
)

unique_questions = all_questions.unique()

print(f"Unique questions to encode: {len(unique_questions):,}")

embeddings = embedding_model.encode(
    unique_questions.tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Map every question to its embedding row
question_to_index = {
    question: index
    for index, question in enumerate(unique_questions)
}


# --------------------------------------------------
# 5. Create symmetric features for question pairs
# --------------------------------------------------
def make_features(question_pairs):
    q1_indices = question_pairs["question1"].map(
        question_to_index
    ).to_numpy(dtype=int)

    q2_indices = question_pairs["question2"].map(
        question_to_index
    ).to_numpy(dtype=int)

    e1 = embeddings[q1_indices]
    e2 = embeddings[q2_indices]

    # Absolute difference captures embedding differences.
    difference = np.abs(e1 - e2)

    # Element-wise product captures shared semantic features.
    product = e1 * e2

    # Normalized embeddings make their dot product cosine similarity.
    cosine = np.sum(product, axis=1, keepdims=True)

    return np.concatenate(
        [difference, product, cosine],
        axis=1
    ).astype(np.float32)


print("\nBuilding training features...")
Xtr = make_features(X_train)

print("Building testing features...")
Xte = make_features(X_test)

print("Feature shape:", Xtr.shape)


# --------------------------------------------------
# 6. Train and evaluate classifiers
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),
    "Linear SVM": LinearSVC(
        random_state=RANDOM_STATE
    )
}

trained_models = {}
results = {}

for name, model in models.items():
    print(f"\n{'=' * 55}")
    print(f"Training {name}...")

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

    trained_models[name] = model
    results[name] = metrics

    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric:>10}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Non-duplicate", "Duplicate"],
            zero_division=0
        )
    )


# --------------------------------------------------
# 7. Save the experiment without overwriting old models
# --------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "models": trained_models,
        "metrics": results,
        "feature_type": "sentence_embeddings_difference_product_cosine"
    },
    OUTPUT_PATH
)

print(f"\nExperiment saved to: {OUTPUT_PATH}")
print("Existing TF-IDF model files were not overwritten.")
