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
for i in range(14):
    cols.append(i)
df = pd.read_csv("Data/train.csv", header=None, names=cols)
df = df.iloc[1:].reset_index(drop=True)

#Select top 5 pca features

X_train = df[range(0,13)].astype(float).values.tolist()
y_train = df[13].astype(int).values.tolist()

df2 = pd.read_csv("Data/test.csv", header=None, names=cols)
df2 = df2.iloc[1:].reset_index(drop=True)

#Select top 5 pca components
X_test = df2[range(0,13)].astype(float).values.tolist()
y_test = df2[13].astype(int).values.tolist()

print("Data points: ", len(X_train))
print("Category 0", y_train.count(0))
print("Category 1", y_train.count(1))
print("Category 2", y_train.count(2))
print("Category 3", y_train.count(3))

counts = [y_train.count(0), y_train.count(1), y_train.count(2), y_train.count(3)]
categories = ["Category 0", "Category 1", "Category 2", "Category 3"]

# Creating the bar graph
plt.bar(categories, counts, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.title('Counts per Category')
plt.show()

#split data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

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

