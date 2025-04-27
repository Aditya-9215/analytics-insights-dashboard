# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score

# Load the data
data = pd.read_csv('Fraud.csv')  # Replace 'data.csv' with your file name

# Step 1: Data Cleaning
# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Fill missing values
data.fillna(0, inplace=True)  # Replace missing values with 0 (or use another strategy)

# Detect and remove outliers
for column in ['amount', 'oldbalanceOrg', 'newbalanceOrig']:
    data = data[(np.abs(data[column] - data[column].mean()) / data[column].std()) < 3]

# Check for multicollinearity
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Step 2: Feature Selection
# Drop unnecessary columns
X = data.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)  # Exclude target and irrelevant columns
y = data['isFraud']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training
# Initialize and train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_val)

# Classification report
print("Classification Report:\n", classification_report(y_val, y_pred))

# Additional metrics
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("ROC AUC Score:", roc_auc_score(y_val, y_pred))

# Step 5: Feature Importance
importance = model.feature_importances_
features = X.columns

# Visualize feature importance
plt.barh(features, importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Fraud Detection Model')
plt.show()