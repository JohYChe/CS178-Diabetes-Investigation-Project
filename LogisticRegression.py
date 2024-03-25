from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np
  
# Fetch dataset 
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

# Data (as pandas dataframes)
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

desired_features = ['gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 
                    'time_in_hospital', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 
                    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 
                    'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 
                    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glipizide', 'glyburide', 
                    'tolazamide', 'insulin', 'change', 'diabetesMed', 'race']

# Filter the dataset to include only the desired features using loc
X = X.loc[:, desired_features]

y = y.values.ravel()
""""
# visualization
plt.figure(figsize=(8, 6))
plt.hist(X['num_medications'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Number of Medication for Diabetes Patients')
plt.xlabel('Number of Medications')
plt.ylabel('Frequency')
plt.show()
"""

# Imputers for string and numeric columns
num_imputer = SimpleImputer(strategy='mean')
str_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputers to handle missing values in diabetes dataset
for column in X.columns:
    if X[column].dtype == 'object':
        X.loc[:, column] = str_imputer.fit_transform(X[[column]]).ravel()
    else:
        X.loc[:, column] = num_imputer.fit_transform(X[[column]]).ravel()

# seed for reproducing results
seed = 1234

# Identifying categorical columns
categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']

# Column transformer, one-hot encoding categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Applying the transformations
X_transformed = preprocessor.fit_transform(X)

# Partition dataset: 20% test, 60% training, 20% validation
X_train_val, X_test, y_train_val, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=seed)


# Finding best Legistic Regression model
def train_and_evaluate(c, p, s):
    lr = LogisticRegression(C=c, penalty=p, solver=s, fit_intercept=True)
    lr.fit(X_train, y_train)

    y_train_pred = lr.predict(X_train)
    y_val_pred = lr.predict(X_val)

    tr_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    return tr_acc, val_acc

results = {}

C = [0.001, 0.01, 0.1, 1, 10]
penalty = ['l1', 'l2']
solver = ['liblinear', 'saga']

for c in C:
    for p in penalty:
        for s in solver:
            tr_acc, val_acc = train_and_evaluate(c, p, s)
            results[c, p, s] = (tr_acc, val_acc)


best_val_acc = float(0)
best_params = None
for params, (tr_acc, val_acc) in results.items():
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = params




# Results Analysis
combined_labels = [f'C={c}, Penalty={p}, Solver={s}' for ((c, p, s), _) in sorted_results]
train_accs = [tr_acc for (_, (tr_acc, _)) in sorted_results]
val_accs = [val_acc for (_, (_, val_acc)) in sorted_results]

# Plotting
x = np.arange(len(combined_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(20, 8))
rects1 = ax.bar(x - width/2, train_accs, width, label='Training Accuracy', color='deepskyblue')
rects2 = ax.bar(x + width/2, val_accs, width, label='Validation Accuracy', color='mediumorchid')

ax.set_xlabel('Logistic Regression Hyperparameter Configurations')
ax.set_ylabel('Accuracy')
ax.set_title('Training and Validation Accuracies with Various Hyperparameters')
ax.set_xticks(x)
ax.set_xticklabels(combined_labels, rotation=45, ha="right")
ax.legend()

fig.tight_layout()
plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
plt.show()

# Evaluate the final model on the test set
best_lr = LogisticRegression(C=best_params[0], penalty=best_params[1], solver=best_params[2], fit_intercept=True)
best_lr.fit(X_train, y_train)
y_pred = best_lr.predict(X_test)

labels = ['<30', '>30', 'NO']
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None, labels=labels)
recall = recall_score(y_test, y_pred, average=None, labels=labels)
f1 = f1_score(y_test, y_pred, average=None, labels=labels)

print('Evaluation metrics on the test set:')
print(f'Class labels: {labels}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
cm_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Logistic Regression')
plt.savefig('output/DT_confmat', bbox_inches='tight')
plt.show()