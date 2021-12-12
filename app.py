import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('loan', 'rb'))


@app.route('/')
def home():
    return render_template('LoanStatus.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['Gender', 'Married', 'Self_Employed', 'ApplicantIncome',
       'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 1:
        return render_template('LoanApproval.html')
    else:
        return render_template('LoanNA.html')


if __name__ == '__main__':
    app.run(debug=True)
    app.run('0.0.0.0', 8000)