from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and encoders
model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))
soil_encoder = pickle.load(open('soil_encoder.pkl', 'rb'))
crop_encoder = pickle.load(open('crop_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])
    soil_type = request.form['Soil_Type']

    # Encode soil type
    soil_type_encoded = soil_encoder.transform([soil_type])[0]

    # Prepare input for prediction
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_type_encoded]])

    # Make prediction
    prediction_encoded = model.predict(input_data)
    prediction = crop_encoder.inverse_transform(prediction_encoded)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)