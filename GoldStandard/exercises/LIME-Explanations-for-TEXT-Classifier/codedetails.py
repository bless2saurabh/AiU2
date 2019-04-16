# Importing packages
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import lime

import sklearn.ensemble
import sklearn.metrics

import sklearn
from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups

vectorizer, rf, newsgroups_train, newsgroups_test, train_vectors, test_vectors, categories=None, None, None, None, None, None, None
# fetch_20newsgroups() connects to online repository to download from (~18000) newsletters on 20 different topics.
# The data is stored as already divided into subsets: 'train' and 'test'
# fetch_20newsgroups() supports fetching the data from individual subsets for specific categories
def set_topics(topics):
    global categories
    categories=topics

def fetch_and_load_training_data():
    global newsgroups_train
    newsgroups_train = fetch_20newsgroups(data_home='.', subset='train', categories=categories)

def fetch_and_load_testing_data():
    global newsgroups_test
    newsgroups_test = fetch_20newsgroups(data_home='.', subset='test', categories=categories)

def print_one_sample_from_training_set(idx):
    print (newsgroups_train.data[idx])

def convert_to_structured_data_form():
    global vectorizer, train_vectors, test_vectors
    # Use tf-idf vectorizer, commonly used for text.
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        lowercase=True, stop_words=text.ENGLISH_STOP_WORDS)

    # Learn vocab, tokenize raw text, encode and create a term-matrix for words in 'train' data
    train_vectors = vectorizer.fit_transform(newsgroups_train.data)

    # tokenize raw text, encode and create a term-matrix for words in 'test' data
    test_vectors = vectorizer.transform(newsgroups_test.data)

def train_model(train_vectors, test_vectors, newsgroups_train):
    global rf
    # Use Random forest for training the model
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    # Provide train data and train labels (target/outcome)
    rf.fit(train_vectors, newsgroups_train.target)

def print_f1_score():
    # Model is trained, make predictions on test data using the trained model.
    pred = rf.predict(test_vectors)

    # Compute F-score i.e., quality metric showing the accuracy level.
    print("F1 score of the RandomForest model: ",
          sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary'))

def combine_vectorizer_and_model_prediction(vectorizer, rf):
    global pipeline
    # Create a pipeline of vectorizer and trained model;
    # The pipeline can take any sample row (newletter text) as input
    # and compute probablities of each category ('atheism', 'christianity') for the input sample
    from sklearn.pipeline import make_pipeline
    pipeline = make_pipeline(vectorizer, rf)

def print_test_sample(idx):
    "Test sample: \n", newsgroups_test.data[idx]

def print_prediction_for_test_sample (idx):
    print('For chosen sample, model prediction:[atheism = %.2f %%, christian = %.2f %%]'
          % (pipeline.predict_proba([newsgroups_test.data[idx]])[0, 0] * 100,
             pipeline.predict_proba([newsgroups_test.data[idx]])[0, 1] * 100))
    print('Actual class: %s'
          % categories[newsgroups_test.target[idx]])

from lime import lime_text
# get an explainer object
from lime.lime_text import LimeTextExplainer
import re

def find_lime_explanations_for_text_sample(idx):
    tokenizer = lambda doc: re.compile(r"(?u)\b\w\w+\b").findall(doc)
    class_names = ['atheism', 'christian']
    # Generate an explanation for it (with at most 6 features
    # i.e., top 6 words deciding the classification)
    exp = LimeTextExplainer(class_names=class_names,
                                  split_expression=tokenizer).explain_instance(newsgroups_test.data[idx],
                                     pipeline.predict_proba, num_features=6)

    # explanation is presented as a list of weighted features.
    # These weighted features are a linear model,
    # which approximates behaviour of the random forest classifier
    # in the vicinity of the test example.
    print('LIME explanations for the sample: ')
    exp.as_list()
    exp.as_pyplot_figure()
    plt.show()
