import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('../dataset/Obesity Classification.csv')

data['Gender'] = data['Gender'].apply(lambda x: 1 if x == "Male" else 0)
column_to_remove = ['ID','Label']
X = data.drop(column_to_remove, axis=1)
y = data['Label']

# X = data.drop(['Id','Species'], axis = 1)
# y = data['Species']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# knn = KNeighborsClassifier(n_neighbors = 12)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# joblib.dump(knn, "knn.pkl")
# with open('datmin/knn.pkl', 'wb') as file:
#     pickle.dump(knn, file)
pickle.dump(knn, open("D:\pawl\datmin/knn.pkl","wb"))

# nb = GaussianNB() 
# nb.fit(X, y)
# joblib.dump(nb, "nb.pkl")

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
# joblib.dump(logreg, "logreg.pkl")
# with open('datmin/logreg.pkl', 'wb') as file:
#     pickle.dump(logreg, file)
pickle.dump(logreg, open("D:\pawl\datmin/logreg.pkl","wb"))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
