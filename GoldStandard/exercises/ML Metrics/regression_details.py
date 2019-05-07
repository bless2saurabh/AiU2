# importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn import datasets, linear_model

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

nones = lambda n: [None for _ in range(n)]
data_frame, X_train, X_test, Y_train, Y_test, model, predicted = nones(7)

def load_input_data_from_file(input_file_path):
    global data_frame
    data_frame = pd.read_csv(input_file_path, header=0)
    print("dataset size: [",data_frame.shape[0],'rows,', data_frame.shape[1],'columns]')
    return (data_frame.head(5))

def do_train_test_split(train_size):
    training_features = ['weight', 'acceleration', 'origin']
    target = 'mpg'

    global X_train, X_test, Y_train, Y_test
    X_train, X_test, Y_train, Y_test = train_test_split(
        data_frame[training_features], data_frame[target], train_size=train_size, random_state=10)
    print("Training dataset size: ", len(X_train))
    print("Testing dataset size: ", len(X_test))

def train_model():
    global model
    model = linear_model.LinearRegression()
    model.fit(X_train,Y_train)

def test_model():
    global predicted
    predicted = model.predict(X_test)

def compute_regression_metrics():
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(Y_test, predicted))
    # Explained R-Square score: 1 is perfect prediction
    print('R Square score: %.2f' % r2_score(Y_test, predicted))
