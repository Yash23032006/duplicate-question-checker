import joblib
from scipy.sparse import hstack

from preprocessing import clean_text

bundle = joblib.load("models/duplicate_model_char.joblib")

q1 = "How do I prepare for the GRE?"
q2 = "How should I plan my preparation for GRE?"

q1 = clean_text(q1)
q2 = clean_text(q2)

print("Cleaned question 1:", q1)
print("Cleaned question 2:", q2)

wv = bundle["word_vectorizer"]
cv = bundle["char_vectorizer"]

w1, w2 = wv.transform([q1]), wv.transform([q2])
c1, c2 = cv.transform([q1]), cv.transform([q2])

word_features = hstack(
    [abs(w1 - w2), w1.multiply(w2)], format="csr"
)
char_features = hstack(
    [abs(c1 - c2), c1.multiply(c2)], format="csr"
)

features = hstack(
    [word_features, char_features], format="csr"
)

model = bundle["models"]["Linear SVM"]

print("Prediction:", model.predict(features)[0])
print("Decision score:", model.decision_function(features)[0])
print("Feature count:", features.shape[1])
print("Expected feature count:", model.n_features_in_)