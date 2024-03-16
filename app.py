import base64
import io
import sys
from django import db
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flaskext.mysql import MySQL
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import ensemble
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request
import requests
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn

# from pushbullet import PushBullet
import joblib
import numpy as np


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change to a strong, secret key
mysql = MySQL(app)

# MySQL configuration
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'ELDERLY'
app.config['MYSQL_DATABASE_HOST'] = '127.0.0.1'  # Modify as needed

mysql.init_app(app)



# Set the title and logo
app.config['APP_TITLE'] = "E"
app.config['APP_LOGO'] = './static/logo.png'

@app.route('/')
def home():
    if 'username' in session:
        return render_template('homepage.html')
    return redirect(url_for('index'))

@app.route('/logout1')
def logout1():
    return redirect(url_for('login'))

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.get_db().cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        account = cursor.fetchone()
        if account:
            session['username'] = username
            flash('Logged in successfully', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Please check your credentials and try again.', 'danger')
    return render_template('login.html', title=app.config['APP_TITLE'], logo=app.config['APP_LOGO'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        phno = request.form['phno']
        cursor = mysql.get_db().cursor()
        cursor.execute("INSERT INTO users (username, password, email,phno) VALUES (%s, %s, %s, %s)", (username, password,email,phno))
        mysql.get_db().commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title=app.config['APP_TITLE'], logo=app.config['APP_LOGO'])


@app.route('/phone-numbers')
def phone_numbers():
    cursor = mysql.get_db().cursor()
    query = "SELECT phno FROM users"
    cursor.execute(query)
    phone_numbers = cursor.fetchall()  # Fetches all phone numbers from the table
    cursor.close()
    
    # Render them in the HTML where 'phone_numbers' is the variable used in your template
    return render_template('Lonelinessgoogleform.html', phone_numbers=phone_numbers)







@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out', 'success')
    return redirect(url_for('login'))

@app.route('/dietrecom')
def diet_recommendation():
    return render_template('Dietrecom.html')

@app.route('/loneliness')
def loneliness():
    return render_template('loneliness.html')

@app.route('/exercise')
def exercise():
    return render_template('Exercise.html')

@app.route('/general')
def general():
    return render_template('GeneralHealth.html')

@app.route('/Lonelinessgoogleform')
def Lonelinessgoogleform():
    return render_template('Lonelinessgoogleform.html')


# Load the trained model (ensure the model is in the same directory as your app)
model = joblib.load('Models/diet_recommendation_random_forest_model.joblib')

# Define the mapping from encoded values back to the original string labels
# This mapping should match the one used during model training
diet_mapping = {
    0: 'Low Carb',
    1: 'Vegan',
    2: 'High Protein',
    3: 'Balanced',
    4: 'High Protein',
    # Add more mappings as needed based on your model's encoding
}

@app.route('/predict_diet', methods=['POST'])
def predict_diet():
    data = request.get_json()

    # Prepare the input features
    # Note: Ensure the encoding matches how the model was trained
    age = int(data['age'])
    gender = 1 if data['gender'] == 'Male' else 0  # Example encoding
    bmi = float(data['bmi'])
    genetic = 0  # You need to define how to encode 'genetic' based on your model's training

    features = np.array([[age, gender, bmi, genetic]])

    # Make prediction
    prediction = model.predict(features)
    diet_recommendation_code = int(prediction[0])  # Convert to Python int

    # Map the prediction code to the corresponding string label
    diet_recommendation = diet_mapping[diet_recommendation_code]

    return jsonify({'diet_recommendation': diet_recommendation})

model_loneliness = joblib.load('Models/loneliness_assessment_random_forest_model.joblib')  # Update this path

marital_status_mapping = {'Single': 0, 'Married': 1, 'Widowed': 2, 'Divorced': 3}
living_situation_mapping = {'Alone': 0, 'With Family': 1, 'With Friends': 2}
social_participation_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

@app.route('/predict_loneliness', methods=['POST'])
def predict_loneliness():
    # Extract data from POST request
    data = request.get_json()
    print(data, file=sys.stderr)
    try:
        # Prepare the feature array with proper encoding
        features = np.array([
            [
                data.get('age', 0),  # Default to 0 if not provided
                1 if data.get('gender') == 'Male' else 0,  # Encode gender as binary
                marital_status_mapping.get(data.get('maritalStatus'), 0),  # Default to 0 if not provided or unknown
                living_situation_mapping.get(data.get('livingSituation'), 0),  # Default to 0 if not provided or unknown
                data.get('socialNetworkSize', 0),  # Default to 0 if not provided
                social_participation_mapping.get(data.get('socialParticipation'), 0)  # Default to 0 if not provided or unknown
            ]
        ])
        
        # Predict using the model
        prediction = model_loneliness.predict(features)
        
        # Assume the model returns a single continuous value for loneliness
        loneliness_score = prediction[0]
        
        # Return the prediction result
        return jsonify({'loneliness_assessment': loneliness_score})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    


# Load the trained model for Exercise Recommendations
model_exercise = joblib.load('Models/exercise_recommendation_rf_model.joblib')

# @app.route('/predict_exercise', methods=['POST'])
# def predict_exercise():
#     data = request.get_json()
#     print(data, file=sys.stderr)
#     try:
#         # Preprocess incoming data
#         processed_data = {
#             'Age': [float(data['age'])],
#             'Gender': [1 if data['gender'].lower() == 'male' else 0],
#             'BMI': [float(data['bmi'])],
#             'Existing Health Conditions': [float(data['existingHealthConditions'])],  # Adjust based on encoding
#             'Mobility and Flexibility': [float(data['mobilityAndFlexibility'])]  # Adjust based on encoding
#         }
#         input_df = pd.DataFrame.from_dict(processed_data)

#         # Predict using the model
#         prediction = model_exercise.predict(input_df)
#         exercise_recommendation = prediction[0]  # Convert prediction to label as needed

#         return jsonify({'exercise_recommendation': exercise_recommendation})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

import sys  # Make sure to import sys for logging

@app.route('/predict_exercise1', methods=['POST'])
def predict_exercise1():
    data = request.get_json()
    print(f"Received data: {data}", file=sys.stderr)  # Log the incoming data

    try:
        # Ensure data preprocessing matches the training phase
        processed_data = {
            'Age': [float(data['age'])],
            'Gender': [1 if data['gender'].lower() == 'male' else 0],
            'BMI': [float(data['bmi'])],
            # Use the correct encoding for categorical features
            'Existing Health Conditions': [data['existingHealthConditions']],  # Adjust as per training
            'Mobility and Flexibility': [data['mobilityAndFlexibility']]  # Adjust as per training
        }
        input_df = pd.DataFrame.from_dict(processed_data)
        print(f"Processed data for prediction: {input_df}", file=sys.stderr)  # Log the processed data

        # Predict using the model
        prediction = model_exercise.predict(input_df)
        exercise_recommendation = prediction[0]  # Assuming direct use of prediction
        print(f"Prediction result: {exercise_recommendation}", file=sys.stderr)  # Log the prediction result

        # Return the prediction result
        return jsonify({'exercise_recommendation': str(exercise_recommendation)})  # Convert to string if needed

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)  # Log the error
        return jsonify({'error': str(e)}), 400



model_general_health = joblib.load('Models/general_health_status_rf_model.joblib')
status_mapping1 = {
    0: 'Healthy',
    1: 'Moderate',
    2: 'Unhealthy'
}
@app.route('/predict_general_health_status', methods=['POST'])
def predict_general_health_status():
    data = request.get_json()
    print(data, file=sys.stderr)
    try:
        # Convert age and bmi to the appropriate numeric types
        age = int(data.get('age', 0))
        bmi = float(data.get('bmi', 0.0))

        # Map 'gender' and 'existingHealthConditions' to their numeric encodings
        gender = 1 if data.get('gender', '').lower() == 'male' else 0
        # Update the health conditions mapping to match your model's training
        health_conditions = {'None': 0, 'Condition1': 1}  # Example mapping
        health_condition = health_conditions.get(data.get('existingHealthConditions', 'None'), 0)

        processed_data = {
            'Age': [age],
            'Gender': [gender],
            'BMI': [bmi],
            'Existing Health Conditions': [health_condition]
        }

        input_df = pd.DataFrame.from_dict(processed_data)
        
        # Make prediction using the model
        prediction = model_general_health.predict(input_df)
        general_health_status = status_mapping1.get(prediction[0], 'Unknown')

        
        # Return the prediction result
        return jsonify({'general_health_status': general_health_status})

    except Exception as e:
        # Log the error and return a message
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400





model_exercise1 = joblib.load('Models/general_health_status_rf_model.joblib')
status_mapping = {
    0: 'Light',
    1: 'Moderate',
    2: 'Intense'
}
@app.route('/predict_exercise', methods=['POST'])
def predict_exercise():
    data = request.get_json()
    print(data, file=sys.stderr)
    try:
        # Convert age and bmi to the appropriate numeric types
        age = int(data.get('age', 0))
        bmi = float(data.get('bmi', 0.0))

        # Map 'gender' and 'existingHealthConditions' to their numeric encodings
        gender = 1 if data.get('gender', '').lower() == 'male' else 0
        # Update the health conditions mapping to match your model's training
        health_conditions = {'None': 0, 'Condition1': 1}  # Example mapping
        health_condition = health_conditions.get(data.get('existingHealthConditions', 'None'), 0)

        processed_data = {
            'Age': [age],
            'Gender': [gender],
            'BMI': [bmi],
            'Existing Health Conditions': [health_condition]
        }

        input_df = pd.DataFrame.from_dict(processed_data)
        
        # Make prediction using the model
        prediction = model_general_health.predict(input_df)
        exercise_recommendation = status_mapping.get(prediction[0], 'Unknown')

        
        # Return the prediction result
        return jsonify({'exercise_recommendation': exercise_recommendation})

    except Exception as e:
        # Log the error and return a message
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400





if __name__ == '__main__':
    app.run(debug=True)
