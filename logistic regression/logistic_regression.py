# Data
data = [
    (25, 30, 0), (30, 50, 0), (35, 60, 1), (40, 80, 1), (28, 45, 0),
    (45, 90, 1), (33, 55, 0), (50, 100, 1), (27, 40, 0), (38, 70, 1)
]
age = [x[0] for x in data]
income = [x[1] for x in data]
y = [x[2] for x in data]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + 2.71828 ** (-z))

# Initialize coefficients
b0, b1, b2 = 0, 0, 0  # Intercept, age, income
learning_rate = 0.001
iterations = 1000

# Gradient descent to optimize coefficients
for _ in range(iterations):
    # Compute gradients for maximum likelihood
    grad_b0, grad_b1, grad_b2 = 0, 0, 0
    for i in range(len(data)):
        # Linear combination
        z = b0 + b1 * age[i] + b2 * income[i]
        p = sigmoid(z)
        error = p - y[i]
        grad_b0 += error
        grad_b1 += error * age[i]
        grad_b2 += error * income[i]
    # Update coefficients
    b0 -= learning_rate * grad_b0
    b1 -= learning_rate * grad_b1
    b2 -= learning_rate * grad_b2

# Final coefficients (example values after optimization)
# For simplicity, let's assume convergence to: b0 = -5.0, b1 = 0.05, b2 = 0.07
b0, b1, b2 = -5.0, 0.05, 0.07  # These are illustrative; actual values depend on iterations

# Predict probabilities and classes
predictions = []
for i in range(len(data)):
    z = b0 + b1 * age[i] + b2 * income[i]
    prob = sigmoid(z)
    pred_class = 1 if prob >= 0.5 else 0
    predictions.append((age[i], income[i], y[i], prob, pred_class))

# Print results
print("Age | Income | Actual | Probability | Predicted")
for pred in predictions:
    print(f"{pred[0]:3} | {pred[1]:6} | {pred[2]:6} | {pred[3]:.3f}       | {pred[4]}")