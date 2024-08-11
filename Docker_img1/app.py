from flask import Flask, request, jsonify

import numpy as np
import joblib
import pandas as pd
from pyngrok import ngrok
import os
import pickle
import os
import glob
import pickle
import joblib
from datetime import datetime
from flask_ngrok import run_with_ngrok
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.metrics._scorer import _PredictScorer
from flask import Flask, render_template, request, jsonify


# Load the models from the files

with open ('./models/final_flight_price_prediction_regression.joblib','rb') as file:
    model=joblib.load(file)

with open ('./models/final_label_encoder.joblib','rb') as file:
    encoder=joblib.load(file)

scaler=joblib.load('./models/final_scaler.joblib')

# Create a function for prediction

def predict_price(input_data, model, label_encoder, scaler):
    text_columns = ['from', 'to', 'flightType', 'agency']
    numerical_columns = ['month', 'speed', 'weekday_num', 'year']

    # Define the speed value here
    predefined_speed = 500  # Example speed value

    # Convert input_data into DataFrame
    df = pd.DataFrame([input_data])

    # Access the best estimator from GridSearchCV
    best_model = model.best_estimator_

    # Encode text-based columns using LabelEncoder
    for column in text_columns:
        if column in df.columns:
            df[column] = label_encoder[column].transform(df[column])

    # Ensure all numerical columns are present
    for column in numerical_columns:
        if column == 'speed':
            df[column] = predefined_speed  # Set the predefined speed value
        elif column in df.columns:
            df[column] = df[column].astype(float)  

    if scaler is not None:
        missing_cols = [col for col in numerical_columns if col in df.columns]
        if missing_cols:
            df[missing_cols] = scaler.transform(df[missing_cols])

    # Make prediction
    prediction = best_model.predict(df)

    return prediction[0]


app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def predict():
    return """

<!DOCTYPE html>
<html>

<head>
    <title>Flight Price Prediction</title>
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f9f9f9;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        h1 {
            color: #39dde0;
            font-size: 36px;
            margin-bottom: 20px;
        }

        form {
            text-align: left;
        }

        input[type="text"],
        input[type="number"] {
            width: 100%;
            padding: 15px;
            margin: 15px 0;
            border: none;
            border-bottom: 2px solid #39dde0;
            font-size: 18px;
            background-color: transparent;
            color: #39dde0;
            transition: border-bottom 0.3s ease;
        }

        input[type="text"]:focus,
        input[type="number"]:focus {
            border-bottom: 2px solid #39dde0;
            outline: none;
        }

        input[type="radio"] {
            margin-right: 10px;
        }

        input[type="submit"] {
            background-color: #39dde0;
            color: #fff;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 20px;
            transition: background-color 0.3s ease;
        }

        input[type="submit"]:hover {
            background-color: #39dde0;
        }

        p#prediction {
            margin-top: 20px;
            font-size: 24px;
            color: #39dde0;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>Flight Price Prediction</h1>
        <hr style="border: 1px solid #39dde0; width: 80%; margin: 20px auto;">
        <form action="/predict" method="POST">

            <label>Departure City:</label><br><br>


            <input type="radio" name="From" value="Aracaju (SE)">Aracaju (SE)<br>
            <input type="radio" name="From" value="Brasilia (DF)">Brasilia (DF)<br>
            <input type="radio" name="From" value="Campo Grande (MS)">Campo Grande (MS)<br>
            <input type="radio" name="From" value="Florianopolis (SC)">Florianopolis (SC)<br>
            <input type="radio" name="From" value="Natal (RN)">Natal (RN)<br>
            <input type="radio" name="From" value="Recife (PE)">Recife (PE)<br>
            <input type="radio" name="From" value="Rio de Janeiro (RJ)">Rio de Janeiro (RJ)<br>
            <input type="radio" name="From" value="Salvador (BH)">Salvador (BH)<br>
            <input type="radio" name="From" value="Sao Paulo (SP)">Sao Paulo (SP)<br><br>

            <hr style="border: 1px solid #39dde0; width: 100%; margin: 20px auto;">

            <label>Destination City:</label><br><br>

            <input type="radio" name="To" value="Aracaju (SE)">Aracaju (SE)<br>
            <input type="radio" name="To" value="Brasilia (DF)">Brasilia (DF)<br>
            <input type="radio" name="To" value="Campo Grande (MS)">Campo Grande (MS)<br>
            <input type="radio" name="To" value="Florianopolis (SC)">Florianopolis (SC)<br>
            <input type="radio" name="To" value="Natal (RN)">Natal (RN)<br>
            <input type="radio" name="To" value="Recife (PE)">Recife (PE)<br>
            <input type="radio" name="To" value="Rio de Janeiro (RJ)">Rio de Janeiro (RJ)<br>
            <input type="radio" name="To" value="Salvador (BH)">Salvador (BH)<br>
            <input type="radio" name="To" value="Sao Paulo (SP)">Sao Paulo (SP)<br><br>

            <hr style="border: 1px solid #39dde0; width: 100%; margin: 20px auto;">

            <label>Flight Type:</label><br><br>

            <input type="radio" name="flightType" value="economic"> Economic<br>
            <input type="radio" name="flightType" value="firstClass"> FirstClass<br>
            <input type="radio" name="flightType" value="premium"> Premium<br><br>

            <hr style="border: 1px solid #39dde0; width: 100%; margin: 20px auto;">


            <label>Agency:</label><br><br>


            <input type="radio" name="agency" value="CloudFy"> CloudFy<br>
            <input type="radio" name="agency" value="FlyingDrops"> FlyingDrops<br>
            <input type="radio" name="agency" value="Rainbow"> Rainbow<br><br>

            <hr style="border: 1px solid #39dde0; width: 100%; margin: 20px auto;">

            <label for="weekday_num">Weekday (0=Sunday, 6=Saturday):</label>
            <input type="number" name="weekday_num" min="0" max="6" placeholder="Day of the week"><br>

            <label for="month">Month:</label>
            <input type="number" name="month" min="1" max="12" placeholder="Month"><br>

            <label for="year">Year:</label>
            <input type="number" name="year" min="2019" max="2123" placeholder="Year"><br>


            <input type="submit" value="Predict">
        </form>
        <p id="prediction"></p>
    </div>
</body>

</html>


    """

@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        From = request.form.get('From')
        To = request.form.get('To')
        flighttype = request.form.get('flightType')
        agency = request.form.get('agency')
        weekday_num = request.form.get('weekday_num')
        month = request.form.get('month')
        year = request.form.get('year')

        # Create a dictionary to store the input data
        data = {
            'from': From,
            'to': To,
            'flightType': flighttype,
            'agency': agency,
            'weekday_num': weekday_num,
            'month': month,
            'year': year
        }

        # Perform prediction using the custom_input dictionary
        prediction = predict_price(data,model,encoder,scaler)
        prediction = str(prediction)

        return jsonify({'Your Flight Price($) will be around': prediction})

# Open a tunnel on the default port 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)