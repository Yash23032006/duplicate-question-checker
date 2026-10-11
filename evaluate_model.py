"""Evaluate the saved duplicate-question models on the same reproducible test split."""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from preprocessing import clean_text

DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/duplicate_model.joblib")
OUTPUT_DIR = Path("evaluation_results")
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42
TEST_SIZE = 0.20


def make_features(vectorizer, question1, question2):
    """Create pair features consistent with train_model.py and app.py."""
    v1 = vectorizer.transform(question1)
    v2 = vectorizer.transform(question2)
    return hstack([abs(v1 - v2), v1.multiply(v2)], format="csr")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. Run python train_model.py first."
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    required = ["question1", "question2", "is_duplicate"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[required].dropna().copy()
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)
    df["question1"] = df["question1"].apply(clean_text)
    df["question2"] = df["question2"].apply(clean_text)
    y = df["is_duplicate"].astype(int)
    X = df[["question1", "question2"]]

    # Split the full pair rows together, preserving pair/label alignment.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    bundle = joblib.load(MODEL_PATH)
    vectorizer = bundle["vectorizer"]
    X_test_features = make_features(
        vectorizer, X_test["question1"], X_test["question2"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Test examples: {len(y_test):,}")
    print(f"Class distribution: {y_test.value_counts().sort_index().to_dict()}")

    for name, model in bundle["models"].items():
        predictions = model.predict(X_test_features)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test_features)[:, 1]
        else:
            scores = model.decision_function(X_test_features)
        metrics["roc_auc"] = roc_auc_score(y_test, scores)

        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        for metric, value in metrics.items():
            print(f"{metric:>10}: {value:.4f}")
        print("\nClassification report:")
        print(classification_report(
            y_test, predictions,
            target_names=["Non-duplicate (0)", "Duplicate (1)"],
            zero_division=0,
        ))

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        print("Confusion matrix (rows=actual, columns=predicted):")
        print(cm)
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Non-duplicate", "Duplicate"],
        )
        display.plot(values_format="d")
        plt.title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        image_path = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"
        plt.savefig(image_path, dpi=160)
        plt.close()
        print(f"Saved confusion matrix: {image_path}")


if __name__ == "__main__":
    main()
