import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def perform_pca(data):
    """
    Perform Principal Component Analysis (PCA) using matrix diagonalization.
    
    Parameters:
    data (numpy.ndarray): The dataset with samples as rows and features as columns.
    
    Returns:
    tuple: Eigenvalues, eigenvectors, and the transformed data in the new basis (principal components).
    """
    # Step 1: Standardize the data (zero mean and unit variance)
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    
    # Step 2: Compute the covariance matrix of the standardized data
    cov_matrix = np.cov(data_std.T)
    
    # Step 3: Perform matrix diagonalization (eigenvalue decomposition)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Step 4: Sort the eigenvalues and eigenvectors by descending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 5: Project the data onto the new basis (principal components)
    principal_components = np.dot(data_std, eigenvectors)
    
    return eigenvalues, eigenvectors, principal_components

# Example usage: Generating a sample 2D dataset
np.random.seed(42)
data = np.random.rand(100, 2)  # 100 samples with 2 features

# Perform PCA
eigenvalues, eigenvectors, transformed_data = perform_pca(data)

# Display results
print("Eigenvalues:\n", eigenvalues)
print("\nEigenvectors (Principal Components):\n", eigenvectors)

# Plot original data and transformed data (first principal component)
plt.figure(figsize=(10, 5))

# Original data plot
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Transformed data plot (projection onto the first principal component)
plt.subplot(1, 2, 2)
plt.scatter(transformed_data[:, 0], np.zeros_like(transformed_data[:, 0]), color='green')
plt.title('Data after PCA (First Principal Component)')
plt.xlabel('Principal Component 1')
plt.ylabel('Zero')

plt.tight_layout()
plt.show()
