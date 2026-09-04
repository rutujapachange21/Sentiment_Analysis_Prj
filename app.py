import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load vectorizer and model
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "modelSent.pkl")

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()

        if review_text:
            # Transform text and predict
            vectorized_input = vectorizer.transform([review_text])
            probabilities = model.predict_proba(vectorized_input)[0]
            predicted_class = model.predict(vectorized_input)[0]

            # Label mapping and confidence extraction
            class_idx = list(model.classes_).index(predicted_class)
            confidence_score = probabilities[class_idx] * 100

            prediction = predicted_class.capitalize()
            confidence = f"{confidence_score:.1f}%"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        review_text=review_text,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
