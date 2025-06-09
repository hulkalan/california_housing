from flask import Flask, request, jsonify,app,url_for,render_template
import joblib

import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler once at startup
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# California housing feature names
FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Validate input keys
    if not all(key in data for key in FEATURE_NAMES):
        return jsonify({"error": f"Missing features. Required: {FEATURE_NAMES}"}), 400

    # Extract features in order
    features = [data[key] for key in FEATURE_NAMES]
    features_array = np.array(features).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features_array)

    # Predict
    prediction = model.predict(features_scaled)[0]

    return jsonify({"predicted_median_house_value": round(prediction, 3)})

if __name__ == "__main__":
    app.run(debug=True)
