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


# fit transform takes a list of each of the cols and return an array contaning the new values
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

#endregion


#region Split Data

# Then we split our data agian into test and train data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

#endregion

#region Training KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=9) # We can specify how many neighbours we need to look for as an argument n_neighbours

model.fit(x_train, y_train) # Input Training data into model

acc = model.score(x_test, y_test) # Find accuracy of model
print('Accuracy: ',acc)

#endregion

#region Testing Our Model

# We create a names list so that we can convert our integer predictions into 
# their string representation 

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

# This will display the predicted class, our data and the actual class
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], '\n', "Data: ", x_test[x], '\n', "Actual: ", names[y_test[x]])
    
    # Now we will we see the neighbors of each point in our testing data
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)

#endregion


