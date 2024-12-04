import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pca import pca

# loading data
data = pd.read_csv('train_after_smote.csv')
#data = data.dropna(subset=['sii'])
target_column = 'sii'
X_train = data.drop(columns=[target_column])
y_train = data[target_column]

# loading test data
data = pd.read_csv('test_after_preprocess.csv')
#data = data.dropna(subset=['sii'])
target_column = 'sii'
X_test = data.drop(columns=[target_column])
y_test = data[target_column]

# scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# initialize PCA MODEL to reduce the data up to the number of components that explains 90% of the variance
model = pca(n_components=0.90)

# Fit PCA on training and testing data
results_train = model.fit_transform(X_train_scaled)
X_train_pca = results_train['PC']

train_pca = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
train_pca['target'] = y_train.reset_index(drop=True)
train_pca.to_csv('train_pca.csv', index=False)

results_test = model.transform(X_test_scaled)
#print(results_test)
X_test_pca = results_test #['PC']
test_pca = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])
#print(y_test)
test_pca['target'] = y_test(drop=True)
test_pca.to_csv('test_pca.csv', index=False)

# Print the top features.
print('Top Features:')
print(results_train['topfeat'])

# Cumulative explained variance
print('Explained Variance:')
print(model.results['explained_var'])

# Explained variance per PC
print('Variance Ratio per PC:')
print(model.results['variance_ratio'])


'''# Reduce the data towards 3 PCs
model = pca(n_components=3)

# Fit transform
results = model.fit_transform(X)

# Print the top features.
print(results['topfeat'])

# Cumulative explained variance
print(model.results['explained_var'])

# Explained variance per PC
print(model.results['variance_ratio'])'''



# Make plot
fig, ax = model.plot()
plt.show()

# 2D plot
fig, ax = model.scatter()
plt.show()

# 3d Plot
fig, ax = model.scatter3d()
plt.show()

# 2D plot
fig, ax = model.biplot(n_feat=4, PC=[0,1])
plt.show()

# 3d Plot
fig, ax = model.biplot3d(n_feat=2, PC=[0,1,2])
plt.show()
