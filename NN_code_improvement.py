#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold, cross_validate,GridSearchCV


df = pd.read_csv('sample.csv', header = None)

# Separate target variable 
x = df.drop([295], axis=1)
y = df[295]

# Check for missing values
print('Check for missing values...')
for c in df.columns:
    missing_value = df[c].isnull().sum()/len(df)*100
    if missing_value > 0:
        print(str(c) + "   " + str(missing_value))
else:
    print("There are no missing values. All good!")


# Low variance filter
print('Removing features with low variance...')
selector = VarianceThreshold(0.05)
x = selector.fit_transform(x)
print('The new datashape is:')
print(x.shape)

x = pd.DataFrame(data=x)

# Encode target variable to numerical (A-E becomes 0-4)
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)


y = pd.DataFrame(data=y)

# Create a new training set
df = pd.concat([x,y], axis=1)

y = df.values[:,48]
y = df.values[:,48]  

# Creating training and test set
print('Splitting the data into training and test data...')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Oversampling under represented classes with SMOTE
print('Oversampling under-represented data...')
sm = SMOTE('not majority')
sm_x_train, sm_y_train = sm.fit_sample(x_train, y_train)
sm_x_test, sm_y_test = sm.fit_sample(x_test, y_test)


# Scale data to feed Neural Network
print('Scaling data...')
scaler = StandardScaler().fit(sm_x_train)

sm_x_train = scaler.transform(sm_x_train)
sm_x_test = scaler.transform(sm_x_test)


# Build NN model for predictions
mlp = MLPClassifier(hidden_layer_sizes= (50,50,50), max_iter=100)

print('Training Neural Network (this could take some time)...')
mlp.fit(sm_x_train, sm_y_train)
y_pred = mlp.predict(sm_x_test)

# Evaluating NN model and visualization
y_pred = y_pred.astype(np.int64)
y_pred = le.inverse_transform(y_pred)

sm_y_test = sm_y_test.astype(np.int64)
sm_y_test = le.inverse_transform(sm_y_test)

cm = confusion_matrix(sm_y_test, y_pred)
cm_norm = cm/cm.astype(np.float).sum(axis=1)
print('Confusion matrix:')
print(cm)
sns.heatmap(cm_norm, center=0.5,
            annot=True, fmt='.2f',
            vmin=0, vmax=1, cmap='Reds',
            xticklabels=['A','B','C','D','E'], 
            yticklabels=['A','B','C','D','E'])
plt.savefig('cm_norm_heatmap.png')
print('Saving normalized confusion matrix heatmap to "cm_norm_heatmap.png"')

print('Classification matrix:')
print(classification_report(sm_y_test,y_pred))




# Show the power of the model with cross validation
print('Cross-validating scores (it will take some time)...')
scores = cross_validate(mlp, sm_x_train, sm_y_train, cv=3)
print('Cross-validated scores: ' + scores)
parameter_space = {'learning_rate_init':[0.0001, 0.01]} 
clf = GridSearchCV(mlp, param_grid=parameter_space, scoring='accuracy', cv=2)
clf.fit(sm_x_train, sm_y_train)

print('Best parameters found: ', clf.best_params_)
print('Best estimator found: ', clf.best_estimator_)
print('Best score found: ',clf.best_score_)

print(mlp.coefs_[4])
# shouldn't have. 