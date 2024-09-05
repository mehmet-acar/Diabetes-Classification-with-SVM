import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score, f1_score, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import scipy.stats as stats

# 1. Load the dataset
# The dataset 'diabetes2.csv' is loaded into a Pandas DataFrame.
data = pd.read_csv('diabetes2.csv')
print(data.head())  # Display the first few rows of the dataset

# 2. Separate features and target variable
# Features are stored in X and the target variable is stored in y.
X = data.drop('Outcome', axis=1)  # Drop the target variable 'Outcome' from the features
y = data['Outcome']  # The target variable

# 3. Split data into training and testing sets
# The dataset is split into training (70%) and testing (30%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Scale the features
# Features are scaled using Min-Max scaling to normalize them between 0 and 1.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Transform the testing data

# 5. Apply SMOTE for balancing the dataset
# SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the training dataset.
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)  # Apply SMOTE to the scaled training data

# 6. Hyperparameter optimization using RandomizedSearchCV
# RandomizedSearchCV is used to find the best hyperparameters for the SVM model.
param_distributions = {
    'C': stats.loguniform(1e-3, 1e3),  # Range of values for the regularization parameter C
    'gamma': stats.loguniform(1e-3, 1e3),  # Range of values for the kernel coefficient gamma
    'kernel': ['linear', 'rbf']  # Types of kernel functions to consider
}
random_search = RandomizedSearchCV(SVC(), param_distributions, n_iter=50, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X_resampled, y_resampled)  # Fit RandomizedSearchCV to the resampled training data

print("RandomizedSearchCV - Best hyperparameters:", random_search.best_params_)  # Print the best hyperparameters
print("RandomizedSearchCV - Best accuracy score:", random_search.best_score_)  # Print the best accuracy score

# 7. Evaluate the best model on training and testing data
best_model = random_search.best_estimator_  # Get the best model from RandomizedSearchCV

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Compute performance metrics
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy score
print("\nClassification Report:\n", classification_report(y_test, y_pred))  # Print classification report

# Confusion Matrix
# Display a heatmap of the confusion matrix.
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
# Compute and plot the ROC curve and AUC score.
decision_function = best_model.decision_function(X_test_scaled)
fpr, tpr, thresholds = roc_curve(y_test, decision_function)  # Compute ROC curve
auc_score = roc_auc_score(y_test, decision_function)  # Compute AUC score

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.', label='SVM (AUC = %0.2f)' % auc_score)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# F1 Score
# Print the F1 score for the model.
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Precision-Recall Curve
# Compute and plot the precision-recall curve.
precision, recall, _ = precision_recall_curve(y_test, decision_function)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
