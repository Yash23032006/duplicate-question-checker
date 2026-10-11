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


@st.cache_resource
def load_bundle():
    return joblib.load("models/duplicate_model.joblib")


st.title("🔍 Duplicate Question Detector")
st.write(
    "Compare two questions using NLP and machine learning."
)

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error(
        "Trained model not found. Run train_model.py first."
    )
    st.stop()


# Automatically select the model
model_name = "Linear SVM"

q1 = st.text_area(
    "Question 1",
    placeholder="How can I learn Python programming?"
)

q2 = st.text_area(
    "Question 2",
    placeholder="What is the best way to start learning Python?"
)


if st.button("Analyze Questions", type="primary"):
    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions.")

    else:
        q1_clean = clean_text(q1)
        q2_clean = clean_text(q2)

        vectorizer = bundle["vectorizer"]
        model = bundle["models"][model_name]

        v1 = vectorizer.transform([q1_clean])
        v2 = vectorizer.transform([q2_clean])

        features = hstack(
            [abs(v1 - v2), v1.multiply(v2)],
            format="csr"
        )

        prediction = int(model.predict(features)[0])
        similarity = cosine_similarity(v1, v2)[0][0]

        st.subheader("Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Cosine Similarity",
                f"{similarity * 100:.2f}%"
            )

        with col2:
            st.metric("Model Used", "Linear SVM")

        if prediction == 1:
            st.success("Prediction: Duplicate Questions ✅")
        else:
            st.warning("Prediction: Non-duplicate Questions")

        score = model.decision_function(features)[0]

        st.metric(
            "SVM Decision Score",
            f"{score:.4f}"
        )

        st.info(
            "Cosine similarity measures text-vector similarity. "
            "The machine learning model predicts whether the "
            "questions are duplicates. Similarity and prediction "
            "are different measures, and neither guarantees "
            "identical meaning."
        )


st.divider()
st.subheader("Model Comparison")

for name, metrics in bundle["metrics"].items():
    st.markdown(f"**{name}**")
    st.write({
        key: round(value, 4)
        for key, value in metrics.items()
    })