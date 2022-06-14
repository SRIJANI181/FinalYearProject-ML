# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 21:08:06 2022

@author: Srijani Das
"""

#Data Pre-processing Step  
#importing libraries  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#importing datasets 
cal1_file = pd.read_csv("cal1.txt", sep='\s+', header=None)

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

#Fitting SVM classifier to the training set
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)  

#Predicting the test set result  
y_pred = svc.predict(x_test)

#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix 
conf = confusion_matrix(y_test, y_pred)

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

#Displaying the matrices
fig,ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6,8))

#confusion matrix
sns.heatmap(conf, annot=True, ax=ax[0]);
ax[0].set_title('confusion matrix')
    
#precision matrix
P = conf/conf.sum(axis=0)
sns.heatmap(P, annot=True, ax=ax[1]);
ax[1].set_title('precision matrix')
    
#recall matrix
R = (conf.T/conf.sum(axis=1)).T
sns.heatmap(R, annot=True, ax=ax[2]);
ax[2].set_title('recall matrix')

#Generating satellite image
#normalising the pixels along each channel 

img = cal1_file.drop(3, axis=1).values
t = ss.transform(img.astype(np.float))

# 512 * 512 = 262144 => total data rows
t_n = np.reshape(t, (512,-1,3))
print(t_n.shape)   
 
fig = plt.figure()
plt.axis('off')
plt.imshow(t_n)
