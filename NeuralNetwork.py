from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
# seed for reproducing results
seed = 1234


# Data imputation and partition
# Fetch dataset 
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
  
# Data (as pandas dataframes) 
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

# Partition dataset: 20% test, 60% training, 20% validation
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=seed)
  
# Imputers for string and numeric columns
num_imputer = SimpleImputer(strategy='mean')
str_imputer = SimpleImputer(strategy='most_frequent')

# Initiate list for categorical columns
categorical_cols = []
numerical_cols = []

# Apply imputers to handle missing values in diabetes dataset
for column in X_train.columns:
    if X_train[column].dtype == 'object':
        X_train.loc[:, column] = str_imputer.fit_transform(X_train[[column]]).ravel()
        categorical_cols.append(column)
    else:
        X_train.loc[:, column] = num_imputer.fit_transform(X_train[[column]]).ravel()
        numerical_cols.append(column)

# Column transformer, one-hot encoding categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Applying the transformations
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# Convert y_train and y_val to 1d arrays
y_train = y_train.values.ravel()
y_val = y_val.values.ravel()
y_test = y_test.values.ravel()

def train_and_evaluate(hls, lr):
    mlp = MLPClassifier(hidden_layer_sizes=hls, learning_rate_init=lr, random_state=seed)
    mlp.fit(X_train, y_train)

    y_train_pred = mlp.predict(X_train)
    y_val_pred = mlp.predict(X_val)

    tr_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    return tr_acc, val_acc

results = {}

hidden_layer_sizes = [(50,), (50, 60), (25, 50, 25), (50, 60, 50), (50, 60, 60), (50, 60, 100)]
learning_rates = [0.05, 0.1]

for lr in learning_rates:
    for hls in hidden_layer_sizes:
        tr_acc, val_acc = train_and_evaluate(hls, lr)
        results[hls, lr] = (tr_acc, val_acc)

best_val_acc = float(0)
best_params = None
for params, (tr_acc, val_acc) in results.items():
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = params

best_mlp = MLPClassifier(hidden_layer_sizes=best_params[0], 
                         learning_rate_init=best_params[1], 
                         random_state=seed)

results_copy = results
sorted_results = sorted(results_copy.items(), key=lambda x: x[1][1])

combined_labels = [f'{hls}, {lr}' for ((hls, lr), _) in sorted_results]
train_accs = [train_acc for (_, (train_acc, _)) in sorted_results]
val_accs = [val_acc for (_, (_, val_acc)) in sorted_results]

x = np.arange(len(combined_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(20,8))
rects_tr = ax.bar(x - width/2, train_accs, width, label='Training Accuracy', color='deepskyblue')
rects_val = ax.bar(x + width/2, val_accs, width, label='Validation Accuracy', color='mediumorchid')

plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

ax.set_xlabel('Neural Network Hyperparameter Configurations (Hidden Layers Config, Learning Rate)')
ax.set_ylabel('Accuracy (Train, Validation)')
ax.set_title('Training and Validation Accuracies on Varying Neural Network Hyperparameters')
ax.set_xticks(x)
ax.set_xticklabels(combined_labels, rotation = -45)
ax.set_ylim(bottom=0.45)
ax.legend()

fig.tight_layout
plt.grid(True)
plt.savefig('output/V2.png', bbox_inches='tight')
plt.show()

# Final Accuracy with Test Data
best_mlp.fit(X_train, y_train)
y_test_pred = best_mlp.predict(X_test)
labels = ['<30', '>30', 'NO']

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average=None, labels=labels)
recall = recall_score(y_test, y_test_pred, average=None, labels=labels)
f1 = f1_score(y_test, y_test_pred, average=None, labels=labels)

print('Evaluation metrics on the test set:')
print(f'Class labels: {labels}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred, labels=labels)
cm_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Feedforward Neural Network')
plt.savefig('output/V2_confmat', bbox_inches='tight')
plt.show()

# How varying amount of training data affects the accuracy
n_tr = [1000, 5000, 10000, 50000, np.shape(X_train)[0]]

tr_err = []
va_err = []

for i in n_tr:
    mlp = best_mlp
    mlp.fit(X_train[:i], y_train[:i])
    tr_err.append(1 - mlp.score(X_train[:i], y_train[:i]))
    va_err.append(1 - mlp.score(X_val, y_val))

plt.figure(figsize=(10, 5))
plt.semilogx(n_tr, tr_err, label="Training Error")
plt.semilogx(n_tr, va_err, label="Testing Error")
plt.title("Effect of Amount of Training Data on Error Rates")
plt.xlabel("Number of training data points")
plt.ylabel("Error rate")
plt.legend()
plt.savefig('output/V1_ntr', bbox_inches='tight')
plt.show()