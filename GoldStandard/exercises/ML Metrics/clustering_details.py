#importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

nones = lambda n: [None for _ in range(n)]
dataset, clustering_cols, clustering_cols_data = nones(3)
colors = ['red', 'blue', 'green', 'cyan', 'orange', 'pink', 'black']
wcss_scale_down_factor = 100000000
def load_input_data_from_file(input_file_path):
    global dataset
    dataset = pd.read_csv(input_file_path)

    print("dataset size: [", dataset.shape[0], 'rows,', dataset.shape[1], 'columns]')
    return dataset.head(5)

def set_cols_used_for_clustering(cols):
    # Column Qty and Rate used for purpose of clustering
    global clustering_cols, clustering_cols_data
    clustering_cols = cols
    clustering_cols_data = dataset.loc[:, clustering_cols]
    print("[", clustering_cols_data.shape[0], ' rows]')
    return clustering_cols_data.head(5)

def deduce_clusters(num_clusters):
    global kmeans, y_kmeans
    # Applying K-means to the data set for two clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # fit_predict() method returns which data points
    # belong to which cluster
    y_kmeans = kmeans.fit_predict(clustering_cols_data)
    visualize_clusters (num_clusters, kmeans, y_kmeans);

def visualize_clusters(num_clusters, kmeans, y_kmeans):
    X = clustering_cols_data.values
    for i in range(num_clusters):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1],
                s=100, c=colors[i], label='Cluster '+str(i+1))
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                s=300, c='yellow', label='Centroids')
    plt.title('Clusters')
    plt.xlabel(clustering_cols[0])
    plt.ylabel(clustering_cols[1])
    plt.legend()
    plt.show()

def print_inertia_metric():
    print ('Inertia is: {0:1.3g}'.format(kmeans.inertia_))

def compute_and_plot_elbow (max_clusters = 6):
    # Finding out optimum number of clusters by
    # "Within Cluster Sum of Squares" (WCSS)
    wcss = []
    X = clustering_cols_data.values
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        random_state=0)
        kmeans.fit(X)

        # Compute WCSS and append it to the list Inertia
        wcss.append(kmeans.inertia_)
        print('#Clusters: ' + str(i) + ' -> Inertia: {0:1.3g}'.format(kmeans.inertia_))

    plt.plot(range(1, max_clusters+1), wcss)  # plotting Elbow method
    plt.title('The Elbow Method')  # naming the title
    plt.xlabel('Number of clusters')  # labeling x axis
    plt.ylabel('WCSS / Inertia')  # labelling the y axis
    plt.show()  # Display the chart