# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 23:46:03 2022

@author: Srijani Das
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from IPython.display import display
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

names = ["KNN", "SVM", "Naive Bayes", "majority voting ensemble"]

models = list()
models.append(('KNN', KNeighborsClassifier(n_neighbors=5, p=2)))
models.append(('SVM', SVC()))
models.append(('NB', GaussianNB()))

classifiers = [
    KNeighborsClassifier(n_neighbors=5, p=2),
    SVC(),
    GaussianNB(),
    VotingClassifier(estimators=models, voting='hard')]

model_cols = []
df=pd.DataFrame(columns=model_cols)
index=0

for name, clf in zip(names, classifiers):
    clf.fit(x_train,y_train)
    df.loc[index,'Classifiers'] = name
    df.loc[index,'Train Accuracy'] = clf.score(x_train,y_train)
    df.loc[index,'Test Accuracy'] = clf.score(x_test,y_test)
    df.loc[index,'Accuracy'] = accuracy_score(y_test,clf.predict(x_test))
    df.loc[index,'Precision'] = precision_score(y_test,clf.predict(x_test), average='micro')
    df.loc[index,'Recall'] = recall_score(y_test,clf.predict(x_test), average='micro')
    df.loc[index,'F1 Score'] = f1_score(y_test,clf.predict(x_test), average='micro')
    df.loc[index,'Kappa Value'] = cohen_kappa_score(y_test,clf.predict(x_test))
    index+=1
    
display(df)
