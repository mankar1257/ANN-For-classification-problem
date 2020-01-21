# -*- coding: utf-8 -*-
"""


@author: Vaibhav R Mankar

"""


#---------------------------------------Data Preprocessing-----------------------------------



# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
file = "/home/vaibhav/Downloads/p020643.psv"
# As this is a PSV file we convert it to CSV 
dataset =pd.read_csv(file,skiprows=1,sep='|').fillna(0)

#  OR you can directy read a csv file 


X = dataset.iloc[:, 0:40].values
y = dataset.iloc[:, 40].values

import matplotlib.pyplot as plt
plt.plot(dataset)
plt.show()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)



 
#-----------------------------------------Making the ANN---------------------------------------

# Importing the Keras libraries and package
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer

# kernel_initializer updates weights
# Activation function - Rectifier

classifier.add(Dense(24, input_dim = 40, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
# Adding Dropout 
classifier.add(Dropout(p = 0.1))


# Adding the second hidden layer
classifier.add(Dense(12, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
# Dropout
classifier.add(Dropout(p = 0.1))

# Adding the third hidden layer
classifier.add(Dense(6, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
# Dropout
classifier.add(Dropout(p = 0.1))
 
# Adding the output layer  
# Activation function - Sigmoid(For aprox value in between 0 and 1 )
classifier.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid' ))

# Using Stochastic Gradient Descent 
# Binary_crossentropy ( as [0,1] classification problem )
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 50)


#----------------------------------------- prediction ---------------------------------------------
 
 
 
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
 
#print(y_pred)

#------------------------------------------Evaluation---------------------------------------------

 
# Keras wrapper and Sci-kit Learn for k-Fold Cross Validation  
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

#Buil_classifier 
def build_classifier():
    
    classifier = Sequential()
    classifier.add(Dense(24, input_dim = 40, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
    classifier.add(Dense(12, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
    classifier.add(Dense(6, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
    classifier.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid' ))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier


# k-Fold cross validator 
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 50)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)




classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 50)
mean = accuracies.mean()
variable = accuracies.std()


#------------------------------------------New-Prediction-----------------------------------------

 
#new_prediction = classifier.predict(np.array([[118,98,38.11,124,81,63,16,0,0,0,0.4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62.29,1,0,0,-0.03,28]]))
# Predicting a single new observation
new_prediction = classifier.predict(np.array([[88,98,0,135,81,64,16,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62.29,1,0,0,-0.03,31]]))
new_prediction = (new_prediction > 0.5)

print(new_prediction)

 