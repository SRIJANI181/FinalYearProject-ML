# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 21:00:45 2022

@author: Srijani Das
"""

#Data Pre-processing Step  
#importing libraries  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#importing datasets 
cal1_file = pd.read_csv("cal1.txt", sep='\s+', header=None)  # separators longer than 1 character and different from '\s+' will be interpreted as regular expressions and will also force the use of the Python parsing engine

#Extracting Independent and dependent Variable

#Independent variables are what we expect will influence dependent variables. 
#A Dependent variable is what happens as a result of the independent variable.
#drop-Remove rows or columns by specifying label names and corresponding axis
x = cal1_file.drop(3, axis=1) #axis{0 or 'index', 1 or 'columns'}, default 0 Whether to drop labels from the index (0 or 'index') or columns (1 or 'columns').
y = cal1_file[3]

# Splitting the dataset into training and test set.

"""Splitting the dataset into training and test set: 
here The pixels values correspond to the independent features while the labels correspond to the dependent or target feature.
We propose the problem as a pixel classification task and split the data into 80/20 split for train and test data. 
Training Set: A subset of dataset to train the machine learning model, and we already know the output.
Test set: A subset of dataset to test the machine learning model, and by using the test set, model predicts the output."""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

#feature Scaling

"""Feature scaling is a method used to normalize the range of independent variables or features of our data"""
from sklearn.preprocessing import StandardScaler
ss = StandardScaler(with_mean=True, with_std=True)
x_train = ss.fit_transform(x_train)    
x_test = ss.transform(x_test) 

#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=5,p=2) 
knn.fit(x_train, y_train) 

#Predicting the test set result  
y_pred = knn.predict(x_test)

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
#'micro':Calculate metrics globally by counting the total true positives, false negatives and false positives.
"""This parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned.
 Otherwise, this determines the type of averaging performed on the data:"""
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
#subplots-This utility wrapper makes it convenient to create common layouts of subplots, including the enclosing figure object, in a single call.

#confusion matrix
sns.heatmap(conf, annot=True, ax=ax[0]); #annot=true If True, write the data value in each cell. heatmap Plot rectangular data as a color-encoded matrix.
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
