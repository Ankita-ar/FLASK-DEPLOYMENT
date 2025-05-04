# Import Libraries
from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('iris_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html') # this is form page

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('index.html', prediction_text = f'Predicted Iris Class: {output}')

if __name__ == '__main__':
    app.run(debug=True)
