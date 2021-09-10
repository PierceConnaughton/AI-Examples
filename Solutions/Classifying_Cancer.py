#region Import Libraries and Data
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# We are going to use premade datasets built in sklearn for this example
cancer = datasets.load_breast_cancer()
print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)

#endregion

#region Split Data

x = cancer.data  # All of the features
y = cancer.target  # All of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# print first 5 instances of data
print(x_train[:5], y_train[:5])

#endregion

#region Training Model

clf = svm.SVC(kernel="linear")
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test) # Predict values for our test data
acc = metrics.accuracy_score(y_test, y_pred) # Test them against our correct values

print('Accuracy: ',acc)

#endregion