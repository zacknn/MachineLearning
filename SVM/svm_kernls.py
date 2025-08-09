import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs

class KernelSVM:
    def __init__(self, kernel='linear', C=1.0, gamma=1.0, degree=3, coef0=1.0, max_iter=1000, tol=1e-3):
        """
        Kernel SVM implementation
        
        Parameters:
        - kernel: 'linear', 'rbf', 'poly', 'sigmoid'
        - C: regularization parameter
        - gamma: kernel coefficient for rbf, poly, sigmoid
        - degree: degree for polynomial kernel
        - coef0: independent term for poly and sigmoid kernels
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        
    def kernel_function(self, X1, X2):
        """Compute kernel matrix between X1 and X2"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        
        elif self.kernel == 'rbf':
            # RBF (Gaussian) kernel: exp(-gamma * ||x1 - x2||^2)
            pairwise_sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                               np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * pairwise_sq_dists)
        
        elif self.kernel == 'poly':
            # Polynomial kernel: (gamma * <x1, x2> + coef0)^degree
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        
        elif self.kernel == 'sigmoid':
            # Sigmoid kernel: tanh(gamma * <x1, x2> + coef0)
            return np.tanh(self.gamma * np.dot(X1, X2.T) + self.coef0)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """Train the SVM using SMO-like algorithm"""
        self.X = X
        self.y = y
        n_samples = X.shape[0]
        
        # Initialize alphas (Lagrange multipliers)
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Precompute kernel matrix
        self.K = self.kernel_function(X, X)
        
        # Training loop (simplified SMO algorithm)
        for iteration in range(self.max_iter):
            alpha_prev = self.alpha.copy()
            
            for i in range(n_samples):
                # Calculate error for sample i
                E_i = self._decision_function_single(i) - y[i]
                
                # Check KKT conditions
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    
                    # Select second alpha randomly
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    E_j = self._decision_function_single(j) - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = max(L, min(H, self.alpha[j]))
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update bias
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
            
            # Check convergence
            if np.linalg.norm(self.alpha - alpha_prev) < self.tol:
                break
        
        # Store support vectors
        support_vector_indices = self.alpha > 1e-5
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.support_vector_alphas = self.alpha[support_vector_indices]
        
        return self
    
    def _decision_function_single(self, i):
        """Decision function for a single training sample"""
        return np.sum(self.alpha * self.y * self.K[:, i]) + self.b
    
    def decision_function(self, X):
        """Compute decision function for input data"""
        K = self.kernel_function(self.X, X)
        return np.dot((self.alpha * self.y), K) + self.b
    
    def predict(self, X):
        """Make predictions"""
        return np.sign(self.decision_function(X))

# Function to create different datasets
def create_datasets():
    datasets = {}
    
    # Linear dataset
    X_linear, y_linear = make_blobs(n_samples=100, centers=2, random_state=42)
    y_linear[y_linear == 0] = -1
    datasets['Linear'] = (X_linear, y_linear)
    
    # Circular dataset (good for RBF)
    X_circles, y_circles = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
    y_circles[y_circles == 0] = -1
    datasets['Circles'] = (X_circles, y_circles)
    
    # Moons dataset (good for RBF/Poly)
    X_moons, y_moons = make_moons(n_samples=100, noise=0.1, random_state=42)
    y_moons[y_moons == 0] = -1
    datasets['Moons'] = (X_moons, y_moons)
    
    return datasets

def plot_svm_results(X, y, svm_model, title):
    """Plot SVM results with decision boundary"""
    plt.figure(figsize=(10, 8))
    
    # Create mesh for decision boundary
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_model.decision_function(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='-', linewidths=2)
    plt.contour(xx, yy, Z, levels=[-1, 1], colors='black', linestyles='--', linewidths=1)
    
    # Plot data points
    colors = ['red' if label == -1 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.8)
    
    # Highlight support vectors
    if hasattr(svm_model, 'support_vectors'):
        plt.scatter(svm_model.support_vectors[:, 0], svm_model.support_vectors[:, 1],
                   s=200, facecolors='none', edgecolors='green', linewidths=2)
    
    plt.title(f'{title}\nSupport Vectors: {len(svm_model.support_vectors) if hasattr(svm_model, "support_vectors") else 0}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)

# Demo: Compare different kernels
def compare_kernels():
    datasets = create_datasets()
    kernels = [
        ('Linear', 'linear', {}),
        ('RBF', 'rbf', {'gamma': 0.5}),
        ('Polynomial', 'poly', {'degree': 3, 'gamma': 0.1}),
        ('Sigmoid', 'sigmoid', {'gamma': 0.01, 'coef0': 1})
    ]
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n=== {dataset_name} Dataset ===")
        plt.figure(figsize=(20, 5))
        
        for i, (kernel_name, kernel_type, params) in enumerate(kernels, 1):
            # Train SVM
            svm = KernelSVM(kernel=kernel_type, C=1.0, **params)
            svm.fit(X, y)
            
            # Calculate accuracy
            predictions = svm.predict(X)
            accuracy = np.mean(predictions == y)
            
            # Plot
            plt.subplot(1, 4, i)
            
            # Create mesh
            h = 0.02
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = svm.decision_function(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot
            plt.contourf(xx, yy, Z, levels=50, alpha=0.4, cmap='RdYlBu')
            plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
            plt.contour(xx, yy, Z, levels=[-1, 1], colors='black', linestyles='--', linewidths=1)
            
            colors = ['red' if label == -1 else 'blue' for label in y]
            plt.scatter(X[:, 0], X[:, 1], c=colors, s=30, alpha=0.8)
            
            if hasattr(svm, 'support_vectors'):
                plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                           s=100, facecolors='none', edgecolors='green', linewidths=2)
            
            plt.title(f'{kernel_name} Kernel\nAccuracy: {accuracy:.2f}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            
            print(f"{kernel_name} Kernel - Accuracy: {accuracy:.2f}, Support Vectors: {len(svm.support_vectors) if hasattr(svm, 'support_vectors') else 0}")
        
        plt.tight_layout()
        plt.show()

# Run the comparison
if __name__ == "__main__":
    print("Comparing different SVM kernels on various datasets...")
    compare_kernels()
    
    # Example of using a specific kernel
    print("\n=== Example: RBF Kernel on Circles Dataset ===")
    datasets = create_datasets()
    X, y = datasets['Circles']
    
    # Train RBF SVM
    rbf_svm = KernelSVM(kernel='rbf', C=1.0, gamma=1.0)
    rbf_svm.fit(X, y)
    
    # Plot results
    plot_svm_results(X, y, rbf_svm, "RBF Kernel SVM on Circles Dataset")
    plt.show()