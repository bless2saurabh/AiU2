import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from imputation_details import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def detect_outliers(dataset, col_name):
    col_data = dataset[col_name]
    col_index = dataset.columns.get_loc(col_name)
    quartile_1, quartile_3 = np.percentile(col_data, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    x=np.where((col_data > upper_bound) | (col_data < lower_bound))
    z=[]
    for i in x:
        y=dataset.iloc[i,col_index]
        z.append(y)
    print ("No.s of outliers: ", str(len(z[0])))

def box_plot (dataset, col_name):
    import seaborn as sns
    sns.boxplot(x=dataset[col_name])