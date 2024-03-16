import pandas as pd
from flask import Flask, request, jsonify
import joblib
from keras.src.saving import load_model

# Corrected import path

app = Flask(__name__)

# Load the trained model and preprocessor
model_path = 'E:/Joc/Hackaton_ai/trained_model.h5'
preprocessor_path = 'E:/Joc/Hackaton_ai/preprocessor.pkl'

model = load_model(model_path)
preprocessor = joblib.load(preprocessor_path)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data_df = pd.DataFrame([data])
    transformed_data = preprocessor.transform(data_df)

    prediction = model.predict(transformed_data)
    occupancy_probability = float(prediction[0][0]) # Convert NumPy float to native Python float

    return jsonify({'occupancy_probability': occupancy_probability})


if __name__ == '__main__':
    app.run(port=8082, host='localhost',debug=True)
