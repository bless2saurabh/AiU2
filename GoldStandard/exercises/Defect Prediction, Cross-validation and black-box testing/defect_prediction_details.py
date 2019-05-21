import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

#Importing packages for machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

x_train,x_test,y_train,y_test,y_pred = None,None,None,None, None

def load_input_data_from_file(input_file_path):
    dataset=pd.read_csv(input_file_path)
    print("dataset size: [", dataset.shape[0], 'rows,', dataset.shape[1], 'columns]')
    return dataset

#Conversion
# Convert the 'bug' column from integer type to boolean ()
# if bug is greater than zero, then, buggy file
def bug_col_boolean_conversion(dataset):
    dataset["bug"]=(dataset["bug"]>0)
    dataset["bug"].head(10)
    return dataset

#Checking for null
def check_for_nulls(dataset):
    null_dataset = dataset[dataset.isnull().any(axis=1)]
    return null_dataset

#Cleanup - Pruning irrelevant columns

def drop_irrelevant_cols(dataset,a,b,c,d):
    x=dataset.drop([a,b,c,d],axis=1) #Dropping Irrelevant input variables
    return(x)
    #y=data_corr["bug"]  #Output variable

#Splitting - split the resultant data into training and testing set
def split_train_test_set (x, y, test_data_fraction):
    global x_train, x_test, y_train, y_test
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_data_fraction,random_state=1000)
    print("Training set: %s rows" %(x_train.shape[0]))
    print("Testing set: %s rows" %(x_test.shape[0]))

def train_model ():
    global y_pred
    log = LogisticRegression(random_state=0)
    log.fit(x_train, y_train)
    y_pred = pd.DataFrame(log.predict(x_test))
    return log

def compute_model_accuracy():
    print ("Accuracy: {:.2%}".format(accuracy_score(y_test,y_pred)))

def compute_confusion_matrix ():
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

def do_k_fold_cross_validation (model, k):
    # k-fold cross validation
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(model, x_train, y_train, cv=k)  # k-fold cross validation
    for fold in scores:
        print("Accuracy: {:.2%}".format(fold))

def do_leave_one_out_cv (model):
    # Leave one out cross validation
    from sklearn import model_selection
    loocv = model_selection.LeaveOneOut()
    results = model_selection.cross_val_score(model, x_train, y_train, cv=loocv)
    for fold in results:
        print("Accuracy: {:.2%}".format(fold))

def save_model(model, filename):
    outfile = open(filename,'wb')
    pickle.dump(model,outfile) #save the model
    outfile.close()
    print("Saved the model as: %s" %(filename))

def load_model (filename):
    import pickle
    infile = open(filename, 'rb')
    new_model = pickle.load(infile)
    infile.close()
    return new_model

def read_test_data(test_data_file):
    ndata = pd.read_csv(test_data_file)
    test_data = ndata.drop(["name", "version", "name.1"], 1)
    return test_data

