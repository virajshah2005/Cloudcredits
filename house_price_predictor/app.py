from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import json

app = Flask(__name__)

# Load feature names from California housing dataset
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# Load saved model and scaler
import os

model_path = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
metrics_path = os.path.join(os.path.dirname(__file__), 'metrics.json')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

try:
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = {'mse': None, 'r2': None}

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/testimonials')
def testimonials():
    return render_template('testimonials.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = []
        for feature in feature_names:
            value = float(request.form.get(feature, 0))
            features.append(value)
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        # Pass mse and r2 from saved metrics
        mse_val = metrics.get('mse')
        r2_val = metrics.get('r2')
        mse_rounded = round(mse_val, 4) if mse_val is not None else 'N/A'
        r2_rounded = round(r2_val, 4) if r2_val is not None else 'N/A'
        return render_template('result.html', prediction=round(prediction, 2), mse=mse_rounded, r2=r2_rounded)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
