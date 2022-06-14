# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 21:05:37 2022

@author: Srijani Das
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import seaborn as sns
sns.set(color_codes=True)

#importing datasets 
cal1_file = pd.read_csv("cal1.txt", sep='\s+', header=None)

cal1_file.head()
cal1_file.info()

print("\nDataset Description : \n")
print(cal1_file.describe())

print("\nClasses : \n")
print(cal1_file[3].value_counts())

#Extracting Independent and dependent Variable
x = cal1_file.drop(3, axis=1)
y = cal1_file[3]
print("\nTotal Number of Datasets : " ,x.shape, y.shape, "\n")

#Separately analysing all the three independent x values/features
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8,4))
sns.boxplot(x[0], ax=ax[0], orient='h')
sns.boxplot(x[1], ax=ax[1], orient='h')
sns.boxplot(x[2], ax=ax[2], orient='h')
plt.show()

#The pixels values corresponding to the independent features while the labels correspond to the 
#dependent or target feature. We propose the problem as a pixel classification task and split the 
#data into 80/20 split for train and test data

# Splitting the dataset into training and test set. 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

print("\nNumber of Datasets for training : " ,x_train.shape)
print("\nNumber of Datasets for testing : " ,x_test.shape, "\n")

#train_idx = x_train.index
#test_idx = x_test.index
#print(test_idx)
#print(train_idx)

#feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler(with_mean=True, with_std=True)
x_train= ss.fit_transform(x_train)    
x_test= ss.transform(x_test) 

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

#sns.barplot( cal1_file[0],cal1_file[1],cal1_file[2],cal1_file[3] )
#t_m=sns.pointplot(cal1_file[0],cal1_file[1],cal1_file[2], hue=cal1_file[3])
#plt.imshow(t_m)

#sns.boxplot(cal1_file['0'], cal1_file['1'],cal1_file['2'], hue=cal1_file['3'])