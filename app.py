import json
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Received API data:", data)
    
    input_array = np.array(list(data.values())).reshape(1, -1)
    new_data = scalar.transform(input_array)
    
    prediction = regmodel.predict(new_data)[0]
    prediction = round(max(prediction, 0), 2)  # Clip negative values
    
    return jsonify({'prediction': prediction})

@app.route('/predict', methods=['POST'])
def predict():
    # Get form input from HTML page
    data = [float(x) for x in request.form.values()]
    
    # Scale input
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print("Final input after scaling:", final_input)
    
    # Predict and format
    prediction = regmodel.predict(final_input)[0]
    prediction = round(max(prediction, 0), 2)  # Prevent negative price
    
    return render_template("home.html", prediction_text=f"The House price prediction is ${prediction}k")

if __name__ == "__main__":
    app.run(debug=True)
