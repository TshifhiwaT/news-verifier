import joblib
from .preprocessing import clean_text

# Load saved model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    prob = model.predict_proba(vec)[0]
    prediction = model.predict(vec)[0]

    return {
        "prediction": "Real" if prediction == 1 else "Fake",
        "confidence": max(prob)
    }

# Example test
if __name__ == "__main__":
    text = "Breaking news: government announces new policy"
    result = predict(text)
    print(result)