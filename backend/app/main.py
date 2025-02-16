from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS



app = Flask(__name__)

CORS(app)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X = np.array([data["humidity"], data["wind_speed"], data["condition_encoded"]])
    prediction = model["intercept"] + np.dot(model["coefficients"], X)
    return jsonify({"predicted_temperature": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
