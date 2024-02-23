import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pydantic import BaseModel
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('cbheart1.pkl', 'rb'))


class Scoringitem(BaseModel):
    Age: int
    Sex: int
    RestingBP: int
    FastingBS: int
    MaxHR: int
    ExerciseAngina: int
    ChestPainType_0: bool
    ChestPainType_1: bool
    ChestPainType_2: bool
    ChestPainType_3: bool
    

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from request.form as floats
        int_features = [x for x in request.form.values()]

# Ensure int_features has exactly 10 values
        if len(int_features) < 10:
    # Pad int_features with default or placeholder values
         default_values = [0.0] * (10 - len(int_features))
        int_features.extend(default_values)

# Now int_features will have exactly 10 values

   
        final_features = pd.DataFrame([int_features], columns=['Age', 'Sex', 'RestingBP', 'FastingBS', 'MaxHR', 'ExerciseAngina','ChestPainType_0', 'ChestPainType_1', 'ChestPainType_2', 'ChestPainType_3'])

        prediction = model.predict(final_features)
        return render_template('home.html', prediction_text="Heart disease: {}".format(prediction[0]))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    input_data = Scoringitem(**data)
    prediction = model.predict([pd.DataFrame(input_data.dict(), index=[0])])
    return jsonify({"prediction": int(prediction[0])})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
