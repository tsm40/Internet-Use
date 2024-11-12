import torch
import pandas as pd

# loading data
data = pd.read_csv('train.csv')
data = data.drop(columns=['id', 'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 'PAQ_C-Season', 'PCIAT-Season', 'SDS-Season', 'PreInt_EduHx-Season'])                  # dropping categorical data
data = data.drop(columns=data.filter(regex="^PCIAT").columns)               # dropping PCIAT question columns
data = data.fillna(data.mean())
data = data.dropna(subset=['sii'])
target_column = 'sii'
X = data.drop(columns=[target_column])
y = data[target_column]
print(X)


X_tensor = torch.tensor(X.values, dtype=torch.float32) # standardize the data manually with PyTorch (mean = 0, std = 1)
mean = X_tensor.mean(dim=0, keepdim=True)
std = X_tensor.std(dim=0, keepdim=True)
X_std = (X_tensor - mean) / std

cov_matrix = torch.mm(X_std.T, X_std) / (X_std.size(0) - 1)
eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)        # eigen decomposition
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real


explained_variance = eigenvalues / eigenvalues.sum()            # normalize eigenvalues to calculate explained variance

sorted_indices = torch.argsort(explained_variance, descending=True)     # sort eigenvalues and calculate cumulative variance
explained_variance_sorted = explained_variance[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]
cumulative_variance = torch.cumsum(explained_variance_sorted, dim=0)

variance_threshold = 0.90                                                               # select components based on the variance threshold
num_components = torch.sum(cumulative_variance <= variance_threshold) + 1
print('NUM COMPONENTS', num_components)
selected_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]

X_pca = torch.mm(X_std, selected_eigenvectors)                              # project data onto the selected components

X_pca_df = pd.DataFrame(X_pca.numpy(), columns=[f'PC{i+1}' for i in range(num_components)])

final_data = pd.concat([X_pca_df, y.reset_index(drop=True)], axis=1)




"""
Visualization
"""

import matplotlib.pyplot as plt


# Select only the top `num_components` columns from `eigenvectors_sorted`
loadings = pd.DataFrame(eigenvectors_sorted[:, :num_components].numpy(), 
                        index=X.columns, 
                        columns=[f'PC{i+1}' for i in range(num_components)])



top_n_features = 5
unique_features = set()  # Set to store unique feature names

for i in range(num_components):  # Iterate through the top 21 PCs
    pc = f'PC{i+1}'
    print(f"\nTop {top_n_features} contributing features to {pc}:")
    top_features = loadings[pc].abs().sort_values(ascending=False).head(top_n_features)
    print(top_features)
    
    # Add these top feature names to the unique_features set
    unique_features.update(top_features.index)

# Convert the set to a sorted list for a consistent display
unique_features_list = sorted(unique_features)
print("\nUnique features contributing to the top 21 PCs:")
print(unique_features_list)
print(len(unique_features_list))



# 1. Explained Variance Plot for Top Components
plt.figure(figsize=(10, 6))
plt.plot(range(1, 22), explained_variance_sorted[:21].numpy(), color="black", marker='o')
for i, (x, y) in enumerate(zip(range(1, 22), explained_variance_sorted[:21].numpy())):
    plt.annotate(
        f"{y:.2f}",
        (x, y),                 
        textcoords="offset points", 
        xytext=(15, 5),          
        ha='center'            
    )
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Top 21 Principal Components')
plt.show()

# 2. Cumulative Variance Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance.numpy(), color="black", marker="o")
plt.xlabel('# of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by # of Components')
plt.axhline(y=0.90, color='cyan', linestyle='dashed', label='90% variance')
plt.annotate('(21, 91.4%)', xy=(21, .90), xytext= (16,.93), color='r')
plt.plot(21, cumulative_variance.numpy()[20], color="r", marker="o")
plt.legend()
plt.show()

print(cumulative_variance.numpy()[20], "cum var")


'''
# 1. Explained Variance Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_sorted) + 1), explained_variance_sorted.numpy(), marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by All Principal Components')
plt.show()

# 2. 2D Scatter Plot of the First Two Principal Components
plt.figure(figsize=(10, 6))
plt.scatter(X_pca_df['PC1'], X_pca_df['PC2'], c=y, cmap='viridis')
plt.colorbar(label='Target Variable (sii)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Scatter Plot of the First Two Principal Components')
plt.show()

# 3. 3D Scatter Plot of the First Three Principal Components (if you kept 3+ components)
from mpl_toolkits.mplot3d import Axes3D

if num_components >= 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca_df['PC1'], X_pca_df['PC2'], X_pca_df['PC3'], c=y, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D Scatter Plot of the First Three Principal Components')
    #plt.show()
'''
