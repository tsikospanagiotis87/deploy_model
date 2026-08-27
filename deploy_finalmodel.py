print('deploy start')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pickle

from flask import Flask
from flask import request
from flask import jsonify

with open('churn_model_C=1.0.bin', 'rb') as file:
    ct, model = pickle.load(file)


app = Flask('churn')

@ app.route('/predict', methods= ['POST'])
def churn_predict():

    customer = request.get_json()
    threshold = customer['threshold']

    if any(isinstance(value, dict) for value in customer.values()):
        customer = pd.DataFrame(customer)
    else:
        customer = pd.DataFrame([customer])

    customer_x = ct.transform(customer)
    customer_churn = {'churn': bool((model.predict_proba(customer_x)[:, 1] > threshold).astype(int)), 'probability': model.predict_proba(customer_x)[0, 1]}

    return jsonify(customer_churn)

if __name__ == '__main__':
    app.run(debug=True, host= 'localhost', port= 8866)


print('deploy finish')