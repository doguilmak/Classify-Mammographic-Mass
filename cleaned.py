# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 23:23:38 2021

@author: doguilmak

dataset: https://www.kaggle.com/overratedgman/mammographic-mass-data-set

"""
#%%
# 1. Importing Libraries

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn.impute import SimpleImputer
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import time
import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Data Preprocessing

# 2.1. Uploading data
start = time.time()
df = pd.read_csv('Cleaned_data.csv')

# 2.2. Plot Gender and Status on Pie Chart
fig = plt.figure(figsize = (12, 12), facecolor='w')
out_df=pd.DataFrame(df.groupby('Severity')['Severity'].count())

patches, texts, autotexts = plt.pie(out_df['Severity'], autopct='%1.1f%%',
                                    textprops={'color': "w"},
                                    startangle=90, shadow=True)

for patch in patches:
    patch.set_path_effects({path_effects.Stroke(linewidth=2.5,
                                                foreground='w')})

plt.legend(labels=['Benign','Malignant'], bbox_to_anchor=(1., .95), title="Failure Type")
plt.show()

# 2.3. Looking for anomalies and duplicated datas
print(df.isnull().sum())
print(list(df.columns))
print("\n", df.head(10))
print("\n", df.describe().T)
print("\n{} duplicated.".format(df.duplicated().sum()))
print(df.info())

# 2.4. Label Encoding
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)

# 2.5. Determination of dependent and independent variables
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
# 3 Artificial Neural Network
"""
# 3.1 Loading Created Model
model = load_model('model.h5')

# 3.2 Checking the Architecture of the Model
model.summary()
"""

model = Sequential()

# 3.3. Adding the input layer and the first hidden layer
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))

# 3.4. Adding the second hidden layer
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

# 3.5. Adding the output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# 3.6. Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# 3.7. Fitting the ANN to the Training set
model_history=model.fit(X_train, y_train, batch_size=200 , epochs = 124, validation_split=0.13)

# 3.8. Predicting the Test set results
y_pred = model.predict(X_test)


# 3.9. Plot accuracy and val_accuracy
print(model_history.history.keys())
"""
model.summary()
model.save('model.h5')
"""
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
plt.show()

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('ANN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()

# 3.10. Confusion Matrix of ANN
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 3.11. Accuracy Score
from sklearn.metrics import accuracy_score
print(f"\nAccuracy score: {accuracy_score(y_test, y_pred)}")

"""
# 3.12. Predicting class
predict = np.array([23.8277,11.8989,2.4393,0.4655,0.2891,11.1013,11.5776,6.8613,35.3166,152.072]).reshape(1, 10)
print(f'Model predicted class as {model.predict_classes(predict)}.')
"""
"""
# 3.13. Plotting model 
from keras.utils import plot_model
plot_model(model, "binary_input_and_output_model.png", show_shapes=True)


from ann_visualizer.visualize import ann_viz
try:
    ann_viz(model, view=True, title="", filename="ann")
except:
    print("PDF saved.")
"""

#%%
# 4 XGBoost

# 4.1 Importing Libraries
from xgboost import XGBClassifier

classifier=XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# 4.2. Building Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_pred, y_test)  # Comparing results
print("\nConfusion Matrix(XGBoost):\n", cm2)

# 4.3. Accuracy Score
from sklearn.metrics import accuracy_score
print(f"\nAccuracy score(XGBoost): {accuracy_score(y_test, y_pred)}")

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
