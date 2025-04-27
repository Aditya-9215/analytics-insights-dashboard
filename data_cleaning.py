# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('Fraud.csv')  # Ensure the dataset file is placed in the directory

# 1. Handle Missing Values
print("Missing values:\n", data.isnull().sum())

# Special handling for 'oldbalanceDest' and 'newbalanceDest' (Merchants starting with 'M')
# Create flags for missing values
data['missing_oldbalanceDest'] = data['oldbalanceDest'].isnull().astype(int)
data['missing_newbalanceDest'] = data['newbalanceDest'].isnull().astype(int)

# Fill missing values with zeros
data.fillna(0, inplace=True)

# 2. Detect and Remove Outliers
# Boxplots to visualize outliers for numerical columns
for column in ['amount', 'oldbalanceOrg', 'newbalanceOrig']:
    plt.boxplot(data[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

# Remove outliers using Z-score
for column in ['amount', 'oldbalanceOrg', 'newbalanceOrig']:
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    data = data[z_scores < 3]  # Retain rows with z-scores less than 3

# 3. Examine Multi-Collinearity
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()