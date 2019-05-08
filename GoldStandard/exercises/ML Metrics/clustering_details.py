#importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

nones = lambda n: [None for _ in range(n)]
dataset, clustering_cols_data, kmeans, y_kmeans = nones(4)

def load_input_data_from_file(input_file_path):
    global dataset
    dataset = pd.read_csv(input_file_path)

    print("dataset size: [", dataset.shape[0], 'rows,', dataset.shape[1], 'columns]')
    return dataset.head(5)

def set_cols_used_for_clustering(col1, col2):
    # Column Qty and Rate used for purpose of clustering
    global clustering_cols_data
    clustering_cols_data = dataset.loc[:, [col1, col2]]
    print("[", clustering_cols_data.shape[0], ' rows]')
    return clustering_cols_data.head(5)

def deduce_clusters(num_clusters):
    global kmeans, y_kmeans
    # Applying K-means to the data set for two clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # fit_predict() method returns which data points
    # belong to which cluster
    y_kmeans = kmeans.fit_predict(clustering_cols_data)
    return y_kmeans

def visualize_clusters():
    X = clustering_cols_data.values
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
                s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],
                s=100, c='blue', label='Cluster 2')
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                s=300, c='yellow', label='Centroids')
    plt.title('Clusters')
    plt.xlabel('Quantity')
    plt.ylabel('Rate')
    plt.legend()
    plt.show()

def print_inertia_metric():
    print ("Interia is", kmeans.inertia_)

def compute_and_plot_elbow (max_clusters = 7):
    # Finding out optimum number of clusters by
    # "Within Cluster Sum of Squares" (WCSS)
    wcss = []
    X = clustering_cols_data.values
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        random_state=0)
        kmeans.fit(X)

        # Compute WCSS and append it to the list Inertia
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 7), wcss)  # plotting Elbow method
    plt.title('The Elbow Method')  # naming the title
    plt.xlabel('Number of clusters')  # labeling x axis
    plt.ylabel('WCSS / Inertia')  # labelling the y axis
    plt.show()  # Display the chart