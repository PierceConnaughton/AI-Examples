# Import files
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#region Read and Trim Data

# Read Data
# Since our data is seperated by semicolons we need to do sep=";"
data = pd.read_csv("E:/College/Semester 7/AI/Data/student/student-mat.csv", sep=";")

#print out first 5 students
print(data.head())

# This trims down our data so we select on the attributes we want to look at 
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#endregion

#region Seperate Data

# We need to choose what attribute we want to predict
# This will be known as the label
# The other attributes that will determine our label are known as features
# We will the create 2 arrays one has the label inside and the other has the features
predict = "G3"
X = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # Labels




# We then split our data into testing and training 
# in this case 90% data will be used to train and the other 10% will test
# We do this so the model is not tested on data it has already seen
# We want to test and train on both labels and features so we will have 4 arrays all together

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

#endregion

#region Implement Linear Regression

# Define Model
linear = linear_model.LinearRegression()

# First Train then Score our models using the arrays we created
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test) # acc stands for accuracy 

#print accuracy
print('Accuracy: ',acc)

#endregion

#region Predicting Specific Students


# Printing out all test data, next to the final grade and the predicted grade
predictions = linear.predict(x_test) # Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#endregion