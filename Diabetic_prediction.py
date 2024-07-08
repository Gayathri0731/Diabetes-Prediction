import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')

X= diabetes_dataset.drop(columns='Outcome',axis=1)
Y= diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y , random_state= 2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

X_train_pred = classifier.predict(X_train)
test_data_accuracy = accuracy_score(X_train_pred, Y_train)

input_data= (5,166,72,19,175,25.8,0.587,51)

input_data_numpy = np.asarray(input_data)
input_data_reshape = input_data_numpy.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)
#print(std_data)

prediction = classifier.predict(std_data)
#print(prediction)

if (prediction[0] == 0):
  print('There is no possible of diabetic')
else:
  print('The person is diabetic')