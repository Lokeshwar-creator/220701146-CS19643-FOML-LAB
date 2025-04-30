# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r'C:\Users\lokes\OneDrive\Desktop\ml project\WINE QUALITY PREDICTION\winequalityN.csv')
print(df.head())

# Basic info and stats
df.info()
print(df.describe().T)

# Handle missing values
print("Missing values before fill:\n", df.isnull().sum())
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())
print("Missing values after fill:", df.isnull().sum().sum())

# Histograms of all columns
df.hist(bins=20, figsize=(10, 10))
plt.tight_layout()
plt.show()

# Average alcohol content by quality
df.groupby('quality')['alcohol'].mean().plot(kind='bar', color='skyblue')
plt.xlabel('Quality')
plt.ylabel('Average Alcohol')
plt.title('Alcohol content by Wine Quality')
plt.show()

# Encode categorical columns
df.replace({'white': 1, 'red': 0}, inplace=True)

# Convert any object type columns to numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Correlation heatmap
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False, cmap='coolwarm')
plt.title("Correlation Matrix (above 0.7)")
plt.show()

# Drop highly correlated feature
df = df.drop('total sulfur dioxide', axis=1)

# Create binary target variable
df['best quality'] = [1 if x > 5 else 0 for x in df['quality']]

# Define features and target
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

# Train-Test Split
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)

# Impute any remaining missing values
imputer = SimpleImputer(strategy='mean')
xtrain = imputer.fit_transform(xtrain)
xtest = imputer.transform(xtest)

# Normalize features
scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Initialize models
models = [LogisticRegression(), XGBClassifier(use_label_encoder=False, eval_metric='logloss'), SVC(kernel='rbf', probability=True)]
model_names = ['Logistic Regression', 'XGBoost', 'SVM (RBF)']

# Train and evaluate models
for i in range(3):
    models[i].fit(xtrain, ytrain)
    train_pred = models[i].predict(xtrain)
    test_pred = models[i].predict(xtest)

    print(f'\n📌 {model_names[i]}')
    print('Training ROC-AUC Score:', metrics.roc_auc_score(ytrain, train_pred))
    print('Validation ROC-AUC Score:', metrics.roc_auc_score(ytest, test_pred))

# Confusion Matrix for best model (XGBoost here)
best_model = models[1]
cm = confusion_matrix(ytest, best_model.predict(xtest))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title("Confusion Matrix - XGBoost")
plt.show()

# Classification report
print("\nClassification Report (XGBoost):")
print(classification_report(ytest, best_model.predict(xtest)))

# Feature importance (XGBoost)
importances = best_model.feature_importances_
feature_names = features.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='green')
plt.title("Feature Importances (XGBoost)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ROC Curve for best model
y_probs = best_model.predict_proba(xtest)[:, 1]
fpr, tpr, thresholds = roc_curve(ytest, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# Train and evaluate models
for i in range(1):
    models[i].fit(xtrain, ytrain)
    train_pred = models[i].predict(xtrain)
    test_pred = models[i].predict(xtest)

    print(f'\n📌 {model_names[i]}')
    print('Training ROC-AUC Score:', metrics.roc_auc_score(ytrain, train_pred))
    print('Validation ROC-AUC Score:', metrics.roc_auc_score(ytest, test_pred))
    
    # ➕ Add classification report
    print("Classification Report:")
    print(classification_report(ytest, test_pred))

  # Classification report for best model (XGBoost)
print("\nClassification Report (XGBoost):")
print(classification_report(ytest, best_model.predict(xtest)))


