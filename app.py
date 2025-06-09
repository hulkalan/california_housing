from flask import Flask, request, jsonify,app,url_for,render_template
import joblib

import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler once at startup
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# California housing feature names



@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    input_data = np.array([[
        float(data['MedInc']),
        float(data['HouseAge']),
        float(data['AveRooms']),
        float(data['AveBedrms']),
        float(data['Population']),
        float(data['AveOccup']),
        float(data['Latitude']),
        float(data['Longitude'])
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if request.is_json:
        return jsonify({"prediction": float(prediction[0])})
    else:
        return f"<h2>Predicted Median House Value: ${prediction[0]:,.2f}</h2>"

if __name__ == "__main__":
    app.run(debug=True)
