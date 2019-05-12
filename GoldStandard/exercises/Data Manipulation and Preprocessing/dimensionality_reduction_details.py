import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imputation_details import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

pca, principalDf = None, None

def drop_cols(dataset, cols):
    return dataset.drop(cols, axis=1)

def convert_discrete_to_numeric_cols(dataset):
    return pd.get_dummies(dataset)

def do_pca (dataset, num_components=4):
    from sklearn.decomposition import PCA
    global pca, principalDf
    pca = PCA(n_components=num_components)
    principalComponents = pca.fit_transform(dataset)

    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2',
                                          'principal component 3', 'principal component 4'])
    print("dataset size: [", principalDf.shape[0], 'rows,', principalDf.shape[1], 'columns]')
    return principalDf.head()

def compute_covariance ():
    from numpy import cov
    # Examine Covariance of deduced principal components
    for i in range(4):
        for j in range(i + 1, 4):
            print('cov({0:d}, {1:d}): {2:.10f}'.format(
                i + 1, j + 1, cov(principalDf['principal component ' + str(i + 1)],
                                  principalDf['principal component ' + str(j + 1)])[0, 1]))

def compute_explained_variances():
    # Component-wise explained variance ratio
    cols = ['% Component-wise explained variance', '% Cumulative explained variance']
    f = lambda l: ['{0:.2f}'.format(100 * x) for x in l]
    return pd.DataFrame({cols[0]: f(pca.explained_variance_ratio_),
                  cols[1]: f(np.cumsum(pca.explained_variance_ratio_))},
                 columns=cols, index=range(1, 5)).rename_axis('Principal components', axis=1)

def plot_component_wise_variance():
    plt.bar(range(1, 5), 100 * pca.explained_variance_ratio_, color='orange', label='Component-wise explained variance')
    plt.plot(range(1, 5), 100 * np.cumsum(pca.explained_variance_ratio_), linewidth=3,
             label='Cumulative explained variance')
    plt.title("Component-wise and Cumulative Explained Variance")
    plt.xlabel('Number of components')
    plt.ylabel('% coverage of data by components')
    plt.legend(loc=7)