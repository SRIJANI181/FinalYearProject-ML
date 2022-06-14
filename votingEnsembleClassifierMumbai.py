# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 00:00:53 2022

@author: Srijani Das
"""

#Data Pre-processing Step  
#importing libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

#importing datasets 
cal1_file = pd.read_csv("bomb.tst", sep='\s+', header=None)

#Extracting Independent and dependent Variable
x = cal1_file.drop(3, axis=1)
y = cal1_file[3]

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

#feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler(with_mean=True, with_std=True)
x_train = ss.fit_transform(x_train)    
x_test = ss.transform(x_test)

# define the base models
models = list()
models.append(('KNN', KNeighborsClassifier(n_neighbors=5, p=2)))
models.append(('SVM', SVC()))
models.append(('NB', GaussianNB()))

# define the hard voting ensemble
ensemble = VotingClassifier(estimators=models, voting='hard')
ensemble.fit(x_train, y_train)



#Predicting the test set result  
y_pred = ensemble.predict(x_test)

#Accuracy score
from sklearn.metrics import accuracy_score 
acc = accuracy_score(y_test, y_pred)
print ("Accuracy : {:.3f}".format(acc))

#Precision Score
from sklearn.metrics import precision_score
prec = precision_score(y_test, y_pred, average='micro')
print("Precision : {:.3f}".format(prec))

#Recall Score
from sklearn.metrics import recall_score
rec = recall_score(y_test, y_pred, average='micro')
print("Recall : {:.3f}".format(rec))

#F1 Score
from sklearn.metrics import f1_score   
f1 = f1_score(y_test, y_pred, average='micro')
print("F1 Score : {:.3f}".format(f1))

#Kappa Value
from sklearn.metrics import cohen_kappa_score
kv = cohen_kappa_score(y_test, y_pred)
print("Kappa Value : {:.3f}".format(kv))