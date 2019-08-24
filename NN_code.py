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
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV


print('Import the data...')
df = pd.read_csv('sample.csv', header = None)

# Separate target variable 
print('Separating target variable...')
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

x = df.values[:,0:48] 
y = df.values[:,48]  

# Creating training and test set
print('Splitting the data into training and test data...')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Oversampling under represented classes with SMOTE
print('Oversampling under-represented data...')
sm = SMOTE('not majority')
sm_x_train, sm_y_train = sm.fit_sample(x_train, y_train)



# Scale data to feed Neural Network
print('Scaling data...')
scaler = StandardScaler().fit(sm_x_train)

sm_x_train = scaler.transform(sm_x_train)
x_test = scaler.transform(x_test)


# Build NN model for predictions
mlp = MLPClassifier(hidden_layer_sizes= (50,50,50), max_iter=100)

print('Training Neural Network (this could take some time)...')
mlp.fit(sm_x_train, sm_y_train)
y_pred = mlp.predict(x_test)

# Evaluating NN model and visualization
y_pred = y_pred.astype(np.int64)
y_pred = le.inverse_transform(y_pred)

y_test = y_test.astype(np.int64)
y_test = le.inverse_transform(y_test)

cm = confusion_matrix(y_test, y_pred)
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
print(classification_report(y_test,y_pred))




# Show the power of the model with cross validation
print('Cross-validating scores (it will take some time)...')
scores = cross_val_score(mlp, sm_x_train, sm_y_train, cv=3)
print('Cross-validated scores: ', scores)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(),scores.std()*2))


# Improving NN model by adjusting the hyperparameters with cross validation
#
# An initial exhaustive grid search with parameter_space = {'hidden_layer_sizes':[(50,50,50),(50,100,50),(100,)],
# 'alpha':[0.05, 0.1, 0.25],'learning_rate_init':[0.0001, 0.0005, 0.001, 0.01]} could not be completed due to the 
# not enough computer power. Therefore, I opted only for the learning rate as parameter.

print('Attempting to find better parameters with GridSearchCV (this will take some time)...')
parameter_space = {'learning_rate_init':[0.0001, 0.01]} 
clf = GridSearchCV(mlp, param_grid=parameter_space, scoring='accuracy', cv=2)
clf.fit(sm_x_train, sm_y_train)

print('Best parameters found: ', clf.best_params_)
print('Best estimator found: ', clf.best_estimator_)
print('Best score found: ',clf.best_score_)

y_pred_clf = clf.predict(x_test)

y_pred_clf = y_pred_clf.astype(np.int64)
y_pred_clf = le.inverse_transform(y_pred_clf)

print('Classification matrix:')
print(classification_report(y_test,y_pred_clf))

# Conclusion: the GridSearchCV result is worse than previous prediction
print("the GridSearchCV's best parameter does not result in better score than my previous prediction using mlp classifier.")
print('The best score overall: ', mlp.best_score_)

