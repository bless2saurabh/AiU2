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

nones = lambda n: [None for _ in range(n)]
categories, \
newsgroups_train_original, newsgroups_test_original, \
train_vectors_original, test_vectors_original, \
newsgroups_train_fixed, newsgroups_test_fixed, \
train_vectors_fixed, test_vectors_fixed, \
vectorizer_original, vectorizer_fixed,\
rf_original, rf_fixed, \
pipeline_original, pipeline_fixed = nones(15)

newsgroups_train = {False:newsgroups_train_original, True: newsgroups_train_fixed}
newsgroups_test = {False:newsgroups_test_original, True: newsgroups_test_fixed}
train_vectors = {False:train_vectors_original, True:train_vectors_fixed}
test_vectors = {False:test_vectors_original, True:test_vectors_fixed}
vectorizer = {False:vectorizer_original, True: vectorizer_fixed}
rf = {False:rf_original, True:rf_fixed}
pipeline = {False:pipeline_original, True:pipeline_fixed}
# fetch_20newsgroups() connects to online repository to download from (~18000) newsletters on 20 different topics.
# The data is stored as already divided into subsets: 'train' and 'test'
# fetch_20newsgroups() supports fetching the data from individual subsets for specific categories
def set_topics(topics):
    global categories
    categories=topics

def fetch_data(remove_included_fields=False):
    if remove_included_fields:
        #global newsgroups_train_fixed, newsgroups_test_fixed
        newsgroups_train[remove_included_fields] = fetch_20newsgroups (subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
        newsgroups_test[remove_included_fields] = fetch_20newsgroups (subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
    else:
        #global newsgroups_train_original, newsgroups_test_original
        newsgroups_train[remove_included_fields] = fetch_20newsgroups(data_home='.', subset='train', categories=categories)
        newsgroups_test[remove_included_fields] = fetch_20newsgroups(data_home='.', subset='test', categories=categories)

def print_test_sample(idx, fixed=False):
    print(newsgroups_test[fixed].data[idx])

def convert_to_structured_data_form(fixed=False):
    vectorizer[fixed] = sklearn.feature_extraction.text.TfidfVectorizer(
            lowercase=True, stop_words=text.ENGLISH_STOP_WORDS)
    # Learn vocab, tokenize raw text, encode and create a term-matrix for words in 'train' data
    train_vectors[fixed] = vectorizer[fixed].fit_transform(newsgroups_train[fixed].data)

    # tokenize raw text, encode and create a term-matrix for words in 'test' data
    test_vectors[fixed] = vectorizer[fixed].transform(newsgroups_test[fixed].data)

def train_model(fixed=False):
    # Provide train data and train labels (target/outcome)
    # Use Random forest for training the model
    rf[fixed] = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    rf[fixed].fit(train_vectors[fixed], newsgroups_train[fixed].target)

    combine_vectorizer_and_model_prediction(fixed)

def combine_vectorizer_and_model_prediction(fixed=False):
    # Create a pipeline of vectorizer and trained model;
    # The pipeline can take any sample row (newletter text) as input
    # and compute probablities of each category ('atheism', 'christianity') for the input sample
    from sklearn.pipeline import make_pipeline
    pipeline[fixed] = make_pipeline(vectorizer[fixed], rf[fixed])


def compute_and_print_f1_score(fixed=False):
    # Model is trained, make predictions on test data using the trained model.
    pred = rf[fixed].predict(test_vectors[fixed])

    # Compute F-score i.e., quality metric showing the accuracy level.
    print("\tF1 score of the model= %.2f %%"
          %(100*sklearn.metrics.f1_score(newsgroups_test[fixed].target, pred, average='binary')))

def print_prediction_for_test_sample (idx, fixed=False):
    print('Model prediction:[atheism = %.2f %%, christian = %.2f %%]'
          % (pipeline[fixed].predict_proba([newsgroups_test[fixed].data[idx]])[0, 0] * 100,
             pipeline[fixed].predict_proba([newsgroups_test[fixed].data[idx]])[0, 1] * 100))

def print_actual_category(idx, fixed=False):
    print('\nActual category: %s' % (categories[newsgroups_test[fixed].target[idx]]))

from lime import lime_text
# get an explainer object
from lime.lime_text import LimeTextExplainer
import re

def find_lime_explanations_for(idx, fixed=False):
    tokenizer = lambda doc: re.compile(r"(?u)\b\w\w+\b").findall(doc)
    class_names = ['atheism', 'christian']
    # Generate an explanation for it (with at most 6 features
    # i.e., top 6 words deciding the classification)
    exp = LimeTextExplainer(class_names=class_names,
                                  split_expression=tokenizer).explain_instance(newsgroups_test[fixed].data[idx],
                                     pipeline[fixed].predict_proba, num_features=6)

    # explanation is presented as a list of weighted features.
    # These weighted features are a linear model,
    # which approximates behaviour of the random forest classifier
    # in the vicinity of the test example.
    print('LIME explanations for the sample: ')
    exp.as_list()
    exp.as_pyplot_figure()
    plt.show()

