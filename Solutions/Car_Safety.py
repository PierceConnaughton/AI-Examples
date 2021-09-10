# Import Librarys
import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Load Car Safety Data
data = pd.read_csv("E:/College/Semester 7/AI/Data/car.data")
print(data.head())  # To check if our data is loaded correctly