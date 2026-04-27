from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved components
model = joblib.load('ada_model.pkl')
scaler = joblib.load('scaler.pkl')
# If you have label encoders for categorical inputs, load them here
# encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Capture top features based on your importance_df: 
        # Respiratory rate, C-Reactive Proteins, Age, etc.
        # For a full implementation, you'd need all 44 features.
        
        try:
            # Extract data from form (mapping to your 44 features)
            # This is a simplified example with the top 3 features
            features = [float(x) for x in request.form.values()]
            final_features = [np.array(features)]
            
            # Scale and Predict
            scaled_features = scaler.transform(final_features)
            prediction = model.predict(scaled_features)
            
            output = "Severe" if prediction[0] == 1 else "Mild/Non-Severe"
            
            return render_template('index.html', prediction_text=f'Predicted Severity: {output}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)