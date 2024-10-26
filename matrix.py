import numpy as np

def matrix_diagonalization(A):
    """
    Perform matrix diagonalization and compute the diagonal matrix and its inverse.
    
    Parameters:
    A (numpy.ndarray): A square matrix to be diagonalized.
    
    Returns:
    tuple: The eigenvectors matrix (P), the diagonal matrix (D), 
           the inverse of the eigenvectors matrix (P_inv), 
           and the inverse of the diagonal matrix (D_inv).
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Form the diagonal matrix D from the eigenvalues
    D = np.diag(eigenvalues)
    
    # Compute the inverse of the matrix of eigenvectors
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    
    # Compute the inverse of the diagonal matrix D (if all eigenvalues are non-zero)
    D_inv = np.diag(1 / eigenvalues)
    
    return P, D, P_inv, D_inv

def get_user_input_matrix():
    """
    Get a square matrix from user input.
    
    Returns:
    numpy.ndarray: A square matrix inputted by the user.
    """
    # Get the size of the matrix
    n = int(input("Enter the size of the square matrix (e.g., 2 for a 2x2 matrix): "))
    
    # Initialize an empty matrix
    matrix = []
    
    # Get the matrix elements from the user
    print(f"Enter the elements of the {n}x{n} matrix row by row (space-separated):")
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        matrix.append(row)
    
    return np.array(matrix)

# Example usage:
A = get_user_input_matrix()
P, D, P_inv, D_inv = matrix_diagonalization(A)

# Display the results
print("\nMatrix A:\n", A)
print("\nMatrix of Eigenvectors (P):\n", P)
print("\nDiagonal Matrix (D):\n", D)
print("\nInverse of P (P_inv):\n", P_inv)
print("\nInverse of Diagonal Matrix (D_inv):\n", D_inv)
