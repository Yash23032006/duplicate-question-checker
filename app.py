import joblib
import streamlit as st
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text


st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="wide"
)


# Load the improved word + character TF-IDF model
@st.cache_resource
def load_bundle():
    return joblib.load("models/duplicate_model_char.joblib")


st.title("🔍 Duplicate Question Detector")
st.write(
    "Compare two questions using NLP and machine learning."
)


# Load the trained model safely
try:
    bundle = load_bundle()

except FileNotFoundError:
    st.error(
        "Improved model not found. "
        "Please run train_model_char.py first."
    )
    st.stop()

except (KeyError, ValueError, EOFError, OSError, ImportError) as error:
    st.error(
        "Could not load the trained model. Check that "
        "models/duplicate_model_char.joblib is present and that "
        "the scikit-learn version matches the version used to train it."
    )
    st.caption(f"Technical details: {error}")
    st.stop()


# Automatically select the Linear SVM model
model_name = "Linear SVM"

if model_name not in bundle["models"]:
    st.error(
        f"Model '{model_name}' was not found in the model bundle."
    )
    st.stop()

model = bundle["models"][model_name]

word_vectorizer = bundle["word_vectorizer"]
char_vectorizer = bundle["char_vectorizer"]


# Input fields
q1 = st.text_area(
    "Question 1",
    placeholder="How can I learn Python programming?"
)

q2 = st.text_area(
    "Question 2",
    placeholder="What is the best way to start learning Python?"
)

MAX_QUESTION_LENGTH = 2000
st.caption(f"Enter two questions to compare. Each question is limited to {MAX_QUESTION_LENGTH:,} characters.")


def create_pair_features(question1, question2):
    """
    Generate pair features using both word and character TF-IDF.

    The feature construction order must match train_model_char.py.
    """

    # Word-level TF-IDF features
    w1 = word_vectorizer.transform([question1])
    w2 = word_vectorizer.transform([question2])

    word_features = hstack(
        [
            abs(w1 - w2),
            w1.multiply(w2)
        ],
        format="csr"
    )

    # Character-level TF-IDF features
    c1 = char_vectorizer.transform([question1])
    c2 = char_vectorizer.transform([question2])

    char_features = hstack(
        [
            abs(c1 - c2),
            c1.multiply(c2)
        ],
        format="csr"
    )

    # Combine features in the same order as training
    combined_features = hstack(
        [word_features, char_features],
        format="csr"
    )

    return combined_features


if st.button("Analyze Questions", type="primary"):

    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions.")

    elif len(q1) > MAX_QUESTION_LENGTH or len(q2) > MAX_QUESTION_LENGTH:
        st.warning(
            f"Each question must be {MAX_QUESTION_LENGTH:,} characters or fewer."
        )

    else:
        try:
            # Apply the same preprocessing used during training
            q1_clean = clean_text(q1)
            q2_clean = clean_text(q2)

            # Generate word + character pair features
            features = create_pair_features(
                q1_clean,
                q2_clean
            )

            # Predict duplicate or non-duplicate
            prediction = int(model.predict(features)[0])

            # Word-level cosine similarity for interpretation
            v1 = word_vectorizer.transform([q1_clean])
            v2 = word_vectorizer.transform([q2_clean])

            similarity = cosine_similarity(v1, v2)[0][0]

            # Decision score from the SVM
            score = float(model.decision_function(features)[0])

            st.subheader("Analysis Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Word Cosine Similarity",
                    f"{similarity * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Model Used",
                    "Linear SVM (Word + Character TF-IDF)"
                )

            if prediction == 1:
                st.success(
                    "Prediction: Duplicate Questions ✅"
                )
            else:
                st.warning(
                    "Prediction: Non-duplicate Questions"
                )

            st.metric(
                "SVM Decision Score",
                f"{score:.4f}"
            )

            st.info(
                "Cosine similarity measures similarity between "
                "word-level TF-IDF vectors. The classifier uses "
                "both word-level and character-level features. "
                "The decision score indicates which side of the "
                "classifier's boundary the pair falls on; it is "
                "not a probability. Neither score guarantees "
                "that two questions have identical meanings."
            )

        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Could not analyze the questions: {error}")


# Display evaluation metrics saved during training
st.divider()
st.subheader("Model Evaluation")

metrics_dict = bundle.get("metrics", {})

if metrics_dict:
    for name, metrics in metrics_dict.items():
        st.markdown(f"**{name}**")
        st.write({
            key: round(float(value), 4)
            for key, value in metrics.items()
        })
else:
    st.info("No evaluation metrics were found in the model bundle.")
