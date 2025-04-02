import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Create an imbalanced dataset
X, y = make_classification(
    n_samples=10000, 
    n_features=10,
    n_informative=5,
    n_redundant=3,
    n_classes=2,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    random_state=42
)

# Check class distribution
print("Original class distribution:")
print(pd.Series(y).value_counts(normalize=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a model on imbalanced data
clf_imbalanced = RandomForestClassifier(random_state=42)
clf_imbalanced.fit(X_train, y_train)
y_pred_imbalanced = clf_imbalanced.predict(X_test)

print("\nPerformance with imbalanced data:")
print(classification_report(y_test, y_pred_imbalanced))

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Check the new class distribution
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_balanced).value_counts(normalize=True))

# Train a model on the balanced data
clf_balanced = RandomForestClassifier(random_state=42)
clf_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_balanced = clf_balanced.predict(X_test)

print("\nPerformance with SMOTE-balanced data:")
print(classification_report(y_test, y_pred_balanced))

# Compare ROC AUC scores
roc_imbalanced = roc_auc_score(y_test, clf_imbalanced.predict_proba(X_test)[:, 1])
roc_balanced = roc_auc_score(y_test, clf_balanced.predict_proba(X_test)[:, 1])

print("\nROC AUC Score - Imbalanced:", roc_imbalanced)
print("ROC AUC Score - SMOTE-balanced:", roc_balanced)

# Visualize confusion matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm_imbalanced = confusion_matrix(y_test, y_pred_imbalanced)
sns.heatmap(cm_imbalanced, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix - Imbalanced')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.subplot(1, 2, 2)
cm_balanced = confusion_matrix(y_test, y_pred_balanced)
sns.heatmap(cm_balanced, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix - SMOTE-balanced')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Feature importance comparison
plt.figure(figsize=(10, 6))
feat_importances_imbalanced = pd.Series(clf_imbalanced.feature_importances_, index=range(10))
feat_importances_balanced = pd.Series(clf_balanced.feature_importances_, index=range(10))

# Create a DataFrame for comparison
feature_importance_df = pd.DataFrame({
    'Imbalanced': feat_importances_imbalanced,
    'SMOTE-balanced': feat_importances_balanced
})

feature_importance_df.plot(kind='bar')
plt.title('Feature Importance Comparison')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.tight_layout()
plt.show()

# Visualize the first two features of the data before and after SMOTE
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
           alpha=0.5, label='Class 0', s=5)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
           alpha=0.5, label='Class 1', s=5)
plt.title('Original Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_train_balanced[y_train_balanced==0, 0], 
           X_train_balanced[y_train_balanced==0, 1], 
           alpha=0.5, label='Class 0', s=5)
plt.scatter(X_train_balanced[y_train_balanced==1, 0], 
           X_train_balanced[y_train_balanced==1, 1], 
           alpha=0.5, label='Class 1', s=5)
plt.title('SMOTE-balanced Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
