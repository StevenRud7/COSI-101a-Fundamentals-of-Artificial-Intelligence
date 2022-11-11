import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

k=11
dftrainX = np.loadtxt("data_train.txt", skiprows=1, usecols=range(1,11))
#print(dftrainX)
dftrainY = np.loadtxt("data_train.txt", skiprows=1, usecols=range(1))
#print(dftrainY)
dftest = np.loadtxt("data_test.txt", skiprows=1)

x = dftrainX.copy()
y = dftrainY.copy()
#print(np.shape(x))
x_te=dftest.copy()


classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(x, y)
#print(np.shape(X_train))
#print(np.shape(y_train))

y_pred_test = classifier.predict(x_te)

print(y_pred_test)
#print(accuracy_score(y, y_pred_test[:400]))

#print(np.shape(y_pred_test))
#print(np.shape(x_te))
classifier2 = KNeighborsClassifier(n_neighbors=k)
classifier2.fit(x_te, y_pred_test)

y_test_test = classifier2.predict(x_te)
#print(y_test_test)
print("The accuracy of the y_test is:",accuracy_score(y_test_test, y_pred_test))


val = []
avg = []

for i in range(1,20):
    for j in range(11):  
        classifier = KNeighborsClassifier(n_neighbors=i)
        classifier.fit(x, y)
        y_pred_test = classifier.predict(x_te)
        classifier2 = KNeighborsClassifier(n_neighbors=i)
        classifier2.fit(x_te, y_pred_test)
        y_test_test = classifier2.predict(x_te)
        val.append(accuracy_score(y_test_test, y_pred_test))
    mean = np.average(val)
    avg.append(mean)
    val = []
print(avg)

with open('Classification Results.txt', 'w') as f:
    for num in y_pred_test:
        num = int(num)
        f.write("%s\n" % num)