from flask import render_template, request, jsonify
from . import app
import pickle
import pandas as pd
import numpy as np

# Load the model
model_filepath = 'src/models/xgbregressor.pkl'
with open(model_filepath, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the preprocessor if it exists
preprocessor_filepath = 'src/models/preprocessor.pkl'
preprocessor = None
try:
    with open(preprocessor_filepath, 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
except FileNotFoundError:
    pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Prepare the input data in the correct format
    input_data = {
        'Year': [data['year']],
        'Kilometers_Driven': [data['kilometers_driven']],
        'Mileage(kmpl)': [data['mileage']],
        'Engine (CC)': [data['engine']],
        'Power (bhp)': [data['power']],
        'Seats': [data['seats']],
        'Automaker': [data['automaker']],
        'Location': [data['location']],
        'Fuel_Type': [data['fuel_type']],
        'Transmission': [data['transmission']],
        'Owner_Type': [data['owner_type']]
    }

    input_df = pd.DataFrame(input_data)

    # Preprocess the input data if preprocessor exists
    if preprocessor:
        input_df = preprocessor.transform(input_df)

    # Make a prediction
    prediction = model.predict(input_df)

    # Convert prediction to a native Python type
    prediction_value = prediction[0].item()

    return jsonify({'prediction': prediction_value})
