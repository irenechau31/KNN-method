# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:35:02 2024

@author: User
"""

import pandas as pd
import numpy as np
import Counter as Counter
#load data
spam_test=pd.read_csv(r'C:\Users\User\OneDrive\Desktop\Fordham\Data Mining (CISC-5790-L02)\HW2_dataset\spam_test.csv')
spam_train=pd.read_csv(r'C:\Users\User\OneDrive\Desktop\Fordham\Data Mining (CISC-5790-L02)\HW2_dataset\spam_train.csv')

#seperate x, y
test_x=spam_test.iloc[:,:-1].values
test_y=spam_test.iloc[:,-1].values
train_x=spam_train.iloc[:,:-1].values
train_y=spam_train.iloc[:,-1].values
k=[1, 5, 11, 21, 41, 61, 81, 101, 201, 401]

def euclidean_dist(x,y):
    dist=np.sqrt(sum((x-y)**2))
    return dist
def knn():
    predictions=[]
    for test_point in test_x:
        for train_point in train_x:
            distance = euclidean_dist(test_x, train_x)
            nearest_neighbors = sorted(distance)[:k]
            for i in nearest_neighbors:
                y_labels=train_y[i]
                most_common=Counter(y_labels).most_common(1)
                predictions.append(most_common)
    return predictions
def accuracy_score():
    pass
def Z_score_normalization():
    pass

