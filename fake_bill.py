#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import joblib  # Assuming you trained a model and saved it

app = Flask(__name__)

# Load your trained model (replace 'model.pkl' with your model file)
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [[
        data["diagonal"],
        data["height_left"],
        data["height_right"],
        data["margin_low"],
        data["margin_up"],
        data["length"]
    ]]
    prediction = model.predict(features)[0]
    return jsonify({"prediction": "Fake" if prediction == 1 else "Real"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

