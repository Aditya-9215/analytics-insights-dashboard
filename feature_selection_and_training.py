# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score

# Load the data
data = pd.read_csv('Fraud.csv')  # Ensure 'Fraud.csv' is in the directory

# Step 1: Variable Selection
# Convert the 'type' column into numerical values using one-hot encoding
X = pd.get_dummies(data.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1), columns=['type'])
y = data['isFraud']  # Target column

# Step 2: Train-Test Split
# Split data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training
# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on validation data
y_pred = model.predict(X_val)

# Step 4: Model Evaluation
# Generate classification report
print("Classification Report:\n", classification_report(y_val, y_pred))

# Compute additional metrics
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("ROC AUC Score:", roc_auc_score(y_val, y_pred))

# Step 5: Feature Importance
# Extract and visualize feature importance
importance = model.feature_importances_
features = X.columns

plt.barh(features, importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Fraud Detection Model')
plt.show()