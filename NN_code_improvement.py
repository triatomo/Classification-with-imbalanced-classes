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
for c in df.columns:
    missing_value = df[c].isnull().sum()/len(df)*100
    if missing_value > 0:
        print(str(c) + "   " + str(missing_value))
else:
    print("There is no missing value. All good!")


# Low variance filter
selector = VarianceThreshold(0.05)
x = selector.fit_transform(x)
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

# Creating training and validation set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Oversampling under represented classes with SMOTE
sm = SMOTE('not majority')
sm_x_train, sm_y_train = sm.fit_sample(x_train, y_train)
sm_x_test, sm_y_test = sm.fit_sample(x_test, y_test)


# Scale data to feed Neural Network
print('Scaling data...')s
scaler = StandardScaler().fit(sm_x_train)

sm_x_train = scaler.transform(sm_x_train)
sm_x_test = scaler.transform(sm_x_test)


# Build NN model for predictions
mlp = MLPClassifier(hidden_layer_sizes= (50,50,50), max_iter=100)

mlp.fit(sm_x_train, sm_y_train)
y_pred = mlp.predict(sm_x_test)


# Evaluate the model
y_pred = y_pred.astype(np.int8)
sm_y_test = sm_y_test.astype(np.int8)

sm_y_test = le.inverse_transform(sm_y_test)
y_pred = le.inverse_transform(y_pred)

cm = confusion_matrix(sm_y_test, y_pred)
print(cm)
sns.heatmap(cm, center=True)
plt.show()
print(classification_report(sm_y_test,y_pred))








# Prediction run 2 with X-validation
scores = cross_val_score(mlp, sm_x_train, sm_y_train, cv=5)
print('Cross-validated scores: ' + scores)




klearn.cross_validation.StratifiedKFold.



print(mlp.coefs_[4])

# Use a single scaler, fit on the train set. 
# It's best to pretend that you are in production, 
# and don't actually have the test dataset. If you 
# fit a separate scaler, you are using information you 
# shouldn't have. 