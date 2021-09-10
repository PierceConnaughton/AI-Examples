#region Import Libraries and Data

# Importing Libraries
import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Load Car Safety Data
data = pd.read_csv("E:/College/Semester 7/AI/Data/car.data") #Car Data has irregular data so we must do some preprossing to change the values to numerical values
print(data.head())  # To check if our data is loaded correctly

#endregion

#region Preprocessing Data

# Label Encoder allows us to change each col into an integer
le = preprocessing.LabelEncoder()


# fit transform takes a list of each of the cola nd return an array contaning the new values
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))


# This recombines data into a feature list and a labels list
X = list(zip(buying, maint, door, persons, lug_boot, safety))  # Features
y = list(cls)  # Labels

# Then we split our data agian into test and train data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

#endregion
