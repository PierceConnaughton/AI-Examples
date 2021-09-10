# Import files
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Read Data
# Since our data is seperated by semicolons we need to do sep=";"
data = pd.read_csv("E:/College/Semester 7/AI/Data/student/student-mat.csv", sep=";")

print(data.head())
