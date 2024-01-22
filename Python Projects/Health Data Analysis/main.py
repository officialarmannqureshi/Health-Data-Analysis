import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Data Collection
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv(url, names=column_names, na_values="?")

# Step 2: Data Exploration and Cleaning
print(data.info())
print(data.describe())

# Handle missing values
data = data.dropna()

# Step 3: Data Analysis
summary_stats = data.describe()

correlation_matrix = data.corr()

# Step 4: Data Visualization with Matplotlib
plt.figure(figsize=(12, 8))

# Plot the distribution of age
plt.subplot(2, 2, 1)
plt.hist(data['age'], bins=20, color='skyblue')
plt.title('Distribution of Age')

# Scatter plot of age vs. cholesterol
plt.subplot(2, 2, 2)
plt.scatter(data['age'], data['chol'], alpha=0.5, color='orange')
plt.title('Scatter Plot: Age vs. Cholesterol')

# Correlation heatmap
plt.subplot(2, 2, 3)
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()
