# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 22:27:17 2021

@author: doguilmak

dataset: https://www.kaggle.com/overratedgman/mammographic-mass-data-set

"""
#%%
# 1. Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Data Preprocessing

# 2.1. Uploading data
start = time.time()
df = pd.read_csv('Cleaned_data.csv')

# 2.2. Looking for anomalies
print(list(df.columns))
print(df.head())
print(df.describe().T)
print("Duplicated: {}".format(df.duplicated().sum()))

# 2.3. Determination of dependent and independent variables
X = df.drop("Severity", axis = 1)
y = df["Severity"]

# 2.6. Splitting test and train 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 2.7. Scaling datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) # Apply the trained

#%%
# Logistic Regression

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(C=2, tol=0.0001)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print('Accuracy of logistic regression classifier: {:.3f}'.format(logr.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#%%
# ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logr.predict_proba(X_test)[:,1])
plt.figure(figsize=(12, 12))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.3f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Logistic Regression')
plt.legend(loc='upper left')
plt.savefig('Log_ROC_curve')
plt.show()

from sklearn.model_selection import cross_val_score

success = cross_val_score(estimator = logr, 
                          X=X_train, 
                          y=y_train, 
                          cv = 4)

# K-Fold Cross Validation
print("\nK-Fold Cross Validation:")
print("Success Mean:\n", success.mean())
print("Success Standard Deviation:\n", success.std())

# Grid Search
from sklearn.model_selection import GridSearchCV
p = [{'tol':[1e-4,1e-3,1e-2], 'C':[1,2,3]},
     {'tol':[1e-4,1e-3,1e-2], 'C':[1,2,3]},
     {'tol':[1e-4,1e-3,1e-2], 'C':[1,2,3]},
     {'tol':[1e-4,1e-3,1e-2], 'C':[1,2,3]}]


gs = GridSearchCV(estimator= logr,
                  param_grid=p,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)

grid_search = gs.fit(X_train, y_train)
best_result = grid_search.best_score_
best_parameters = grid_search.best_params_
print("\nGrid Search:")
print("Best result:\n", best_result)
print("Best parameters:\n", best_parameters)

#%%
# Saving Model
"""
import pickle
file = "logr.save"
pickle.dump(logr, open(file, 'wb'))

downloaded_data = pickle.load(open(file, 'rb'))
print(downloaded_data.predict(X_test))
"""

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
