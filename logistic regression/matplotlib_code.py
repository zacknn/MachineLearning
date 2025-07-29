import matplotlib.pyplot as plt
import numpy as np

# Data
age = [25, 30, 35, 40, 28, 45, 33, 50, 27, 38]
income = [30, 50, 60, 80, 45, 90, 55, 100, 40, 70]
buy = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# Logistic regression coefficients
b0, b1, b2 = -5.0, 0.05, 0.07

# Create scatter plot
plt.figure(figsize=(8, 6))
for i in range(len(age)):
    if buy[i] == 1:
        plt.scatter(age[i], income[i], c='green', label='Buy (1)' if i == 2 else "", s=100)
    else:
        plt.scatter(age[i], income[i], c='red', label='Not Buy (0)' if i == 0 else "", s=100)

# Plot decision boundary
# Decision boundary: b0 + b1*age + b2*income = 0
# income = (-b0 - b1*age) / b2
age_range = np.linspace(20, 55, 100)
income_boundary = (-b0 - b1 * age_range) / b2
plt.plot(age_range, income_boundary, 'b-', label='Decision Boundary')

# Shade regions
# Region where p > 0.5 (Buy predicted)
plt.fill_between(age_range, income_boundary, 110, color='green', alpha=0.1, label='Predict Buy')
# Region where p < 0.5 (Not Buy predicted)
plt.fill_between(age_range, 20, income_boundary, color='red', alpha=0.1, label='Predict Not Buy')

# Customize plot
plt.xlabel('Age (years)')
plt.ylabel('Income ($1000s)')
plt.title('Logistic Regression: Customer Purchase Prediction')
plt.legend()
plt.grid(True)
plt.xlim(20, 55)
plt.ylim(20, 110)
plt.show()