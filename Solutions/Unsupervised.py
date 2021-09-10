#region Import Libraries and Data
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Load data from sklearn
digits = load_digits()

# Scale data down so features are put into a range between -1 and 1 for simpler calculations
data = scale(digits.data)
y = digits.target

# Define number of clusters
k = 10

# Define how many samples and features we have by getting the data shape
samples, features = data.shape

print(data.shape)

#endregion

#region Score Our Model

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

#endregion



