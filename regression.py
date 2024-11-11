import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#Preprocess data
cols = []
for i in range(82):
    cols.append(i)
df = pd.read_csv("Data/train.csv", header=None, names = cols)
df = df.iloc[1:].reset_index(drop=True)

#Select top 5 pca features
trainCols = [78, 41, 11, 17, 80]
df = df = df.dropna(subset=[df.columns[78], df.columns[41], df.columns[81], df.columns[11], df.columns[17],df.columns[80]])
label = 81

X = df[trainCols].astype(float).values.tolist()
y = df[label].astype(int).values.tolist()

print("Data points: ", len(X))
print("Category 0", y.count(0))
print("Category 1", y.count(1))
print("Category 2", y.count(2))
print("Category 3", y.count(3))

#split data
X_train, X_test, y_train, y_test = train_test_split(X[1:], y[1:], test_size=0.25, random_state=16)

#Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=200, random_state=16)

logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

#Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

