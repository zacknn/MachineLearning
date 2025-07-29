import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dataset
X = np.array([
    [1000, 2, 10],
    [1200, 2, 15],
    [1500, 3, 8],
    [1600, 3, 20],
    [1800, 4, 5],
    [2000, 4, 12],
    [2100, 3, 25],
    [2300, 5, 7],
    [2500, 4, 10],
    [2700, 5, 15]
])
y = np.array([300, 340, 420, 410, 490, 510, 500, 580, 610, 660])

# Normalize features
X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
scale_factors = X.max(axis=0) - X.min(axis=0)
offsets = X.min(axis=0)

# Initialize parameters
w = np.zeros(3)
b = 0.0
learning_rate = 0.01
iterations = 1000
n = len(y)

# Gradient descent
for _ in range(iterations):
    y_pred = X_scaled @ w + b
    loss = np.mean((y - y_pred) ** 2)
    dw = -(2/n) * (X_scaled.T @ (y - y_pred))
    db = -(2/n) * np.sum(y - y_pred)
    w -= learning_rate * dw
    b -= learning_rate * db

# Unscale weights
w_original = w / scale_factors
b_original = b - np.sum(w * offsets / scale_factors)

print(f"Learned model: y = {w_original[0]:.4f}x1 + {w_original[1]:.4f}x2 + {w_original[2]:.4f}x3 + {b_original:.4f}")
print(f"Final MSE: {loss:.2f}")

# Predict new house: 2200 sq ft, 4 bedrooms, 10 years
x_new = np.array([2200, 4, 10])
x_new_scaled = (x_new - offsets) / scale_factors
y_new = x_new_scaled @ w + b
print(f"Predicted price for 2200 sq ft, 4 bedrooms, 10 years: ${y_new:.0f}k")

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of data points
ax.scatter(X[:, 0], X[:, 1], y, c='blue', marker='o', label='Data Points')

# Create meshgrid for the regression plane (size and bedrooms)
x1_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 10)
x2_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 10)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Compute predicted prices for the plane, fixing x3 (age) at mean value
mean_age = np.mean(X[:, 2])  # Average age
x3_fixed = (mean_age - offsets[2]) / scale_factors[2]
y_grid = (w[0] * (x1_grid - offsets[0]) / scale_factors[0] +
          w[1] * (x2_grid - offsets[1]) / scale_factors[1] +
          w[2] * x3_fixed + b)

# Plot the regression plane
ax.plot_surface(x1_grid, x2_grid, y_grid, color='red', alpha=0.5, label='Regression Plane')

# Labels and title
ax.set_xlabel('Size (sq ft)')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price ($1000s)')
plt.title('Multiple Linear Regression: House Prices')
plt.show()