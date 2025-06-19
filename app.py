from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import ollama
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("opioid_risk_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)
CORS(app)  # Allow requests from the Flutter frontend

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Opioid Risk Prediction API is running"}), 200

# API Endpoint for opioid risk prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        doctor_notes = data.get("doctor_notes", "").strip()

        if not doctor_notes:
            return jsonify({"error": "Doctor notes are required"}), 400

        # Convert text into TF-IDF features
        input_features = vectorizer.transform([doctor_notes])

        # Predict risk
        prediction = model.predict(input_features)[0]
        confidence = np.max(model.predict_proba(input_features)) * 100  # Convert to percentage

        # Risk assessment message
        risk_message = (
            f"🔴 High Risk of Opioid Addiction (Confidence: {confidence:.2f}%)"
            if prediction == 1
            else f"🟢 Low Risk of Opioid Addiction (Confidence: {confidence:.2f}%)"
        )

        # Ollama AI response
        ollama_response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": f"Assess opioid addiction risk based on: {doctor_notes}"}]
        )

        return jsonify({
            "risk_prediction": risk_message,
            "ai_analysis": ollama_response["message"]["content"]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
