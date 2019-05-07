import numpy as np # linear algebra
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import logging,sys

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

nones = lambda n: [None for _ in range(n)]
data_frame, X_train, X_test, Y_train, Y_test, model, predictions, cM = nones(8)
extracted_dataset = []
target = []
def load_input_data_from_file (input_file_path):
    global data_frame
    data_frame = pd.read_csv(input_file_path, header=0)
    print("dataset size: [",data_frame.shape[0],'rows,', data_frame.shape[1],'columns]')
    return (data_frame.head(5))

def strip_target_column():
    global extracted_dataset, target
    dataset = np.array(data_frame)
    # Sepratating out target column from rest of data-set columns
    for row in dataset:
        extracted_dataset.append(row[1:])
        target.append(row[0])

def do_train_test_split(test_size=0.2):
    # Splitting independent data and targeted data as test and train
    global X_train, X_test, Y_train, Y_test
    X_train, X_test, Y_train, Y_test = train_test_split(extracted_dataset,
                                                        target, test_size=test_size,
                                                        random_state=100)

    print("Training dataset size: ", len(X_train))
    print("Testing dataset size: ", len(X_test))

def train_model ():
    # Train the model
    global model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

def test_model ():
    # predicting test data values for clf_entropy model
    global predictions
    predictions = model.predict(X_test)

def compute_confusion_matrix ():
    from sklearn.metrics import confusion_matrix
    global cM
    cM = confusion_matrix(Y_test, predictions)
    print(cM)

def compute_classification_metrics():
    tp, fp, fn, tn = cM.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print("Recall/Sensitivity = {:.2%}".format(tp / (tp + fn)))
    print("Specificity = {:.2%}".format(tn / (tn + fp)))
    print("Accuracy = {:.2%}".format((tp + tn) / (tp + tn + fp + fn)))
    print("Precision = {:.2%}".format(tp / (tp + fp)))
    f1score = 2 * (recall * precision) / (precision + recall)
    print("f1 score = {:.2%}".format(f1score))