%%writefile app.py
from flask import Flask, jsonify, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_regression_model.joblib')

# Get the feature columns from the training data, excluding the 'Year' column itself
# and assuming CountryCode_ features start after 'Year'
import numpy as np
# To correctly handle the one-hot encoding for new data, we need the full list of columns
# that the model was trained on. We can reconstruct this from X_train.
# The first column is 'Year', the rest are one-hot encoded CountryCodes.
model_columns = ['Year'] + [col for col in X_train.columns if col != 'Year']

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    year = data['Year']
    country_code = data['CountryCode']

    # Create a DataFrame for the input with all model columns initialized to 0/False
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0, 'Year'] = year
    
    # Initialize all CountryCode columns to False (or 0) for the new prediction
    for col in model_columns:
        if 'CountryCode_' in col:
            input_df.loc[0, col] = False

    # Set the specific CountryCode column to True
    country_col_name = f'CountryCode_{country_code}'
    if country_col_name in input_df.columns:
        input_df.loc[0, country_col_name] = True
    else:
        # Handle unseen country codes if necessary, e.g., by logging or returning an error
        print(f"Warning: CountryCode '{country_code}' not seen during training.")
        # For now, we'll proceed with all CountryCode columns as False, which effectively
        # makes it 'unspecified' if the one-hot column doesn't exist.

    # Convert Year to the correct data type if needed, and ensure boolean types for OHE columns
    input_df['Year'] = input_df['Year'].astype(int)
    for col in model_columns:
        if 'CountryCode_' in col:
            input_df[col] = input_df[col].astype(bool)

    # Make prediction
    prediction = model.predict(input_df[model_columns])
    prediction_proba = model.predict_proba(input_df[model_columns])

    # Return the prediction as JSON
    # 'HighUnemployment': 1 if prediction is high, 0 if low
    response = {
        'Year': year,
        'CountryCode': country_code,
        'HighUnemployment_Prediction': int(prediction[0]),
        'Probability_LowUnemployment': prediction_proba[0][0],
        'Probability_HighUnemployment': prediction_proba[0][1]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
