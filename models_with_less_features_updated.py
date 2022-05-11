import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

features = pd.read_csv("less_features_updated.csv")
print(features.shape)
studentInfo = pd.read_csv("OULAD/studentInfo.csv")
studentInfo = studentInfo.drop_duplicates('id_student')
y = studentInfo[['final_result']]
#y['final_result'] = y['final_result'].map({'Pass': 2, 'Fail': 1, 'Withdrawn':0, 'Distinction':3}) # mapping to int
print(y)
print(y.shape)

print(features.columns[1:2])
x= features[features.columns[1:2]] #taking gender = 'M'
print(x.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train,y_train)
predictions = forest.predict(X_test)
print("Accuracy when subject has gender  = M:",metrics.accuracy_score(y_test, predictions))

x= features[features.columns[2:3]] #taking gender = 'F'
#print(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train,y_train)
predictions = forest.predict(X_test)
print("Accuracy when subject has gender = F:",metrics.accuracy_score(y_test, predictions))

x= features[features.columns[5:6]] #taking disability flag = 'T'
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train,y_train)
predictions = forest.predict(X_test)
print("Accuracy when subject has disability flag = T:",metrics.accuracy_score(y_test, predictions))

x= features[features.columns[9:10]] #taking MAX(studentInfo.studied_credits)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train,y_train)
predictions = forest.predict(X_test)
print("Accuracy based on max credits studied:",metrics.accuracy_score(y_test, predictions))

x= features[features.columns[8:9]] #taking MAX(studentInfo.num_of_prev_attempts)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train,y_train)
predictions = forest.predict(X_test)
print("Accuracy based on max number of previous attempts:",metrics.accuracy_score(y_test, predictions))

x= features[features.columns[166:167]] #taking MAX(studentVle.sum_click)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train,y_train)
predictions = forest.predict(X_test)
print("Accuracy based on max number clicks:",metrics.accuracy_score(y_test, predictions))