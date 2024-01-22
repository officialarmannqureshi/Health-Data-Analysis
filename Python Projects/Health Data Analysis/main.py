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
plt.figure(figsize=(16, 10))

# Plot the distribution of age
plt.subplot(2, 3, 1)
plt.hist(data['age'], bins=20, color='skyblue')
plt.title('Distribution of Age')

# Scatter plot of age vs. cholesterol
plt.subplot(2, 3, 2)
plt.scatter(data['age'], data['chol'], alpha=0.5, color='orange')
plt.title('Scatter Plot: Age vs. Cholesterol')

# Correlation heatmap
plt.subplot(2, 3, 3)
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation Heatmap')

# Health Risk Score Calculation
weights = {'age': 0.2, 'trestbps': 0.1, 'chol': 0.1, 'thalach': -0.2, 'oldpeak': 0.3}
data['health_risk_score'] = data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].mul(weights).sum(axis=1)

# Plot the Health Risk Score distribution
plt.subplot(2, 3, 4)
plt.hist(data['health_risk_score'], bins=20, color='salmon')
plt.title('Health Risk Score Distribution')

# Scatter plot of Health Risk Score vs. Age
plt.subplot(2, 3, 5)
plt.scatter(data['age'], data['health_risk_score'], alpha=0.5, color='purple')
plt.title('Scatter Plot: Age vs. Health Risk Score')

# Find the age and number of persons with the maximum cholesterol levels
max_chol_age = data.loc[data['chol'].idxmax(), 'age']
max_chol_count = data[data['chol'] == data['chol'].max()].shape[0]
print(f"The age with the maximum number of persons suffering from high cholesterol is {max_chol_age} with {max_chol_count} persons.")

# Create a histogram of age distribution for people with the target condition
plt.subplot(2, 3, 6)
plt.hist(data[data['target'] == 1]['age'], bins=20, color='green', alpha=0.7)
plt.title('Age Distribution of People with Target Condition')

# Calculate the percentage of people in each age group with high cholesterol
chol_age_groups = pd.cut(data['age'], bins=range(20, 100, 10))
percentage_chol_by_age_group = (
    data[data['chol'] > data['chol'].mean()]
    .groupby(chol_age_groups, observed=False)
    .size() / data.groupby(chol_age_groups, observed=False).size() * 100
)

# Find the age group with the maximum percentage of people facing high cholesterol
max_chol_age_group = percentage_chol_by_age_group.idxmax()

# Convert max_chol_age_group to numeric value (midpoint of the interval)
max_chol_age_group_midpoint = max_chol_age_group.mid

# Plot the percentage of people with high cholesterol in each age group
fig, ax = plt.subplots(figsize=(10, 5))
percentage_chol_by_age_group.plot(kind='bar', color='red', alpha=0.7)
ax.set_xlabel('Age Group')
ax.set_ylabel('Percentage of People with High Cholesterol')
ax.set_title('Percentage of People with High Cholesterol in Each Age Group')
plt.axvline(x=max_chol_age_group_midpoint, color='black', linestyle='--', linewidth=2, label='Max Percentage Age Group')
plt.legend()

plt.tight_layout()
plt.show()
